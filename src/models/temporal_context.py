from abc import abstractmethod
from itertools import product
import random
import numpy as np
import math
from src.utils import set_seed


@abstractmethod
class TemporalContext:
    """A class that tells how to allocate a fixed budget of frames into 
    a context to give to a generative model. The allocation is done as follows:
     - in the past (t<=0), several possibilities are given by several tuples, 
       e.g. (-3,-2,-1,0) or (-6,-4,-2,0)  
     - in the future (t>0), same thing, 
       e.g. (1,2,3,4) or (2,4,6,8). """
    def __init__(self, past_frames: int, future_frames: int):
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.past_horizon = None
        self.future_horizon = None

    @abstractmethod
    def sample_template(self, seed: int):
        raise NotImplementedError("This method should be implemented by the subclass")
    
    @abstractmethod
    def sample_mask(self, seed: int):
        raise NotImplementedError("This method should be implemented by the subclass")


class UniformContext(TemporalContext):
    def __init__(self, past_frames: int, future_frames: int):
        super().__init__(past_frames, future_frames)
        self.past_horizon = self.past_frames - 1 
        self.future_horizon = self.future_frames
        self.template = tuple(range(-self.past_horizon, self.future_horizon+1))
        self.mask = np.array(self.template) <= 0
        self.actions = [(0, 0, self.template, self.mask)]

    def sample_template(self, seed: int):
        return self.template

    def sample_mask(self, seed: int):
        return self.mask


class LongRangeFixedPastContext(TemporalContext):
    def __init__(self, past_frames: int, future_frames: int, future_horizon: int):
        super().__init__(past_frames, future_frames)
        assert past_frames % 2 == 0, "Past frames must be even"
        assert future_frames >= past_frames//2, "Future frames must be at least half of past frames"
        self.future_horizon = future_horizon
        self.past_horizon = past_frames - 1
        self.templates, self.mask = self.assemble_templates()
        self.actions = [
            (tt_idx, 0, template, self.mask) 
            for tt_idx, template in enumerate(self.templates)
        ]

    def assemble_templates(self):
        n_templates = self.future_horizon // self.future_frames

        templates = []

        # first template: uniform around present
        templates.append(tuple(range(-self.past_frames+1, self.future_frames+1)))
        # then go across future 
        present_idces = tuple(range(-self.past_frames//2+1, 1))
        future_idces = tuple(range(1, self.future_frames+1))
        for _ in range(n_templates-1):
            # shift the future idces to predict
            future_idces = tuple(x+self.future_frames for x in future_idces)
            first_future = future_idces[0]
            # future idces taken as context
            future_context = tuple(range(first_future-self.past_frames//2, first_future))
            # template is present + future context + future idces
            template = (*present_idces, *future_context, *future_idces)
            template = tuple(x for x in template)
            templates.append(template)

        # in that case, the conditionning is simple
        mask = np.array(templates[0]) <= 0  # True True ... False False 

        return templates, mask
    
    def sample_template(self, seed: int):
        with set_seed(seed, backend='random'):
            return random.choice(self.templates)
        
    def sample_mask(self, seed: int):
        with set_seed(seed, backend='random'):
            return self.mask
        

class Hierarchy2Context(TemporalContext):
    def __init__(self, past_frames: int, future_frames: int):
        super().__init__(past_frames, future_frames)
        # TODO: horrible hardcoding of the templates used in their paper
        assert past_frames == 4 and future_frames == 3, "Hierarchy2Context only supports past_frames=4 and future_frames=3 for the moment"
        self.past_horizon = 9
        self.future_horizon = 21
        self.templates, self.masks = self.assemble_templates()
        self.used_masks = {i: self.masks[i] for i in range(len(self.templates))}
        self.actions = [
            (0, 0, self.templates[0], self.masks[0]),
            (1, 0, self.templates[1], self.masks[1]),
            (2, 0, self.templates[2], self.masks[2]),
            (3, 0, self.templates[3], self.masks[3]),
            (4, 0, self.templates[4], self.masks[4]),
            (5, 0, self.templates[5], self.masks[5]),
            (6, 0, self.templates[6], self.masks[6]),
        ]

    def assemble_templates(self):
        templates = [
            (-9, -6, -3, 0, 1, 10, 20),
            (0, 1, 2, 3, 4, 10, 20),
            (3, 4, 5, 6, 7, 10, 20),
            (6, 7, 8, 9, 10, 11, 20),
            (9, 10, 11, 12, 13, 14, 20),
            (12, 13, 14, 15, 16, 17, 20),
            (15, 16, 17, 18, 19, 20, 21),
        ]
        masks = [
            np.array([True, True, True, True, False, False, False]),
            np.array([True, True, False, False, False, True, True]),
            np.array([True, True, False, False, False, True, True]),
            np.array([True, True, False, False, True, False, True]),
            np.array([True, True, True, False, False, False, True]),
            np.array([True, True, True, False, False, False, True]),
            np.array([True, True, True, False, False, True, False]),
        ]

        return templates, masks
    
    def sample_template(self, seed: int):
        with set_seed(seed, backend='random'):
            idx = random.choice(list(self.used_masks.keys()))
            return self.templates[idx]

    def sample_mask(self, seed: int):
        with set_seed(seed, backend='random'):
            idx = random.choice(list(self.used_masks.keys()))
            return self.used_masks[idx]


class MultiscaleSymmetricalContext(TemporalContext):
    def __init__(self, 
        past_frames: int, 
        past_horizon: float = 9,
        flexible: bool = False
    ):
        super().__init__(past_frames, past_frames-1)
        self.past_horizon = self.future_horizon = past_horizon
        self.possible_templates = self.assemble_temporal_templates()
        self.used_masks, self.actions = self.get_inference_strategy(self.possible_templates)  # parameters for the inference strategy
        # now refine the templates and their masks to only the ones used in the inference strategy
        self.templates = {i: self.possible_templates[i] for i, v in self.used_masks.items() if v != []}
        self.used_masks = {i: v for i, v in self.used_masks.items() if v != []}
        self.flexible = flexible

    @staticmethod
    def alpha_from_horizon(frames, horizon):
        """Invert the relation alpha -> horizon"""
        poly = [1] + [0]*(frames-2) + [-horizon-0.25, horizon-0.75]
        roots = np.roots(poly)
        alpha = roots.real[np.abs(roots.imag) < 1e-10].max().item()
        # checks
        horizon_real = (alpha ** frames - 1) / (alpha - 1)
        assert int(horizon_real) == horizon, f"Computed horizon {horizon_real} does not match target horizon {horizon}"
        return alpha

    def assemble_temporal_templates(self):
        """Create all the (unique) templates for the given alpha range. """
        # get unique templates
        templates = set()
        horizon = 0
        for horizon in range(self.future_frames, self.future_horizon+1):
            alpha = self.alpha_from_horizon(self.future_frames, horizon)
            past_template = tuple(-int((alpha**n-1) / (alpha-1)) for n in range(self.past_frames-1,-1,-1))
            future_template = tuple(sorted(-x for x in past_template if x < 0))
            horizon = max(horizon, -min(past_template), max(future_template))
            templates.add((*past_template, *future_template))
        templates = list(templates)
        self.past_horizon = self.future_horizon = horizon
        # now order the templates (lexicographically on the future)
        templates_future = [tuple(x for x in t if x > 0) for t in templates]
        sorted_indices = sorted(range(len(templates_future)), key=lambda i: (len(templates_future[i]), templates_future[i]))
        templates = [templates[i] for i in sorted_indices]
        return templates
    
    def sample_template(self, seed: int):
        with set_seed(seed, backend='random'):
            idx = random.choice(list(self.used_masks.keys()))
            return self.templates[idx]

    def sample_mask(self, seed: int):
        with set_seed(seed, backend='random'):
            if self.flexible:
                mask = np.array([True] * self.past_frames + [False] * self.future_frames)
                np.random.shuffle(mask)
                return mask
            else:
                idx = random.choice(list(self.used_masks.keys()))
                return random.choice(self.used_masks[idx])

    def get_inference_strategy(self, templates):
        horizon = self.future_horizon
        used_masks = {i: [] for i in range(len(templates))}
        completed = np.array([i<=0 for i in range(-2*horizon, 2*horizon+1)])
        central_idx = 2*horizon
        actions = []  # template applied at this step, its center and its mask
        count = 0
        while not all(completed[central_idx:central_idx+horizon+1]) and count < 100:
            count += 1
            # go from the largest template to the smallest one 
            current_action = None
            min_steps_left_alone = np.inf
            for tt_idx, shift in product(range(len(templates)-1, -1, -1), range(horizon+1)):
                template = np.array(templates[tt_idx])
                # reject templates longer than the horizon to cover
                if template[-1] > horizon:
                    continue
                # shift the template
                template_shifted = template + central_idx + shift
                # check overlap with already generated steps
                overlap = completed[template_shifted].sum().item()
                # check the previous gen steps that are not covered by the template
                # if too many, this will cause time inconsistency
                gen_steps = np.where(completed)[0]
                gen_steps = gen_steps[gen_steps>central_idx]
                steps_left_alone = np.setdiff1d(gen_steps, template_shifted)
                # now privilege templates which anchors on the very last future
                anchor_last = (template + shift)[-1] == horizon
                if overlap == 4:  # admissible template
                    if anchor_last:  # no hesitation: choose this template
                        min_steps_left_alone = steps_left_alone.size
                        mask = completed[template_shifted]
                        current_action = (tt_idx, shift, template, mask)
                        break
                    elif (steps_left_alone.size < min_steps_left_alone or
                          steps_left_alone.size == min_steps_left_alone):  # choose the template that conditions maximally on its own generations
                        min_steps_left_alone = steps_left_alone.size
                        mask = completed[template_shifted]
                        current_action = (tt_idx, shift, template, mask)
            # complete future frames
            completed_copy = completed.copy()
            completed_copy[template_shifted] = True

            tt_idx, selected_shift, selected_template, selected_mask = current_action
            completed[selected_template + central_idx + selected_shift] = True
            actions.append(current_action)
            used_masks[tt_idx].append(selected_mask)

        if count == 100:
            raise ValueError("Could not find an inference strategy.")
        
        return used_masks, actions


if __name__ == "__main__":

    # template = MultiscaleFixedPastContext(past_frames=4, future_frames=3, alpha=2.0)
    # template = MultiscaleSymmetricalContext(past_frames=4, alpha_min=1.1, alpha_max=2.3, flexible=True)
    # template = UniformContext(past_frames=4, future_frames=3)
    # template = LongRangeFixedPastContext(past_frames=4, future_frames=3, future_horizon=15)
    template = Hierarchy2Context(past_frames=4, future_frames=3)
    print(template.sample_template(0))
    print(template.sample_mask(0))