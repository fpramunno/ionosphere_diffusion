#!/usr/bin/env python3
import subprocess
import re

def get_reservations():
    output = subprocess.check_output(["scontrol", "show", "reservation"], text=True)
    blocks = output.strip().split("\n\n")

    for block in blocks:
        name = re.search(r"ReservationName=(\S+)", block).group(1)
        start = re.search(r"StartTime=(\S+)", block).group(1)
        end = re.search(r"EndTime=(\S+)", block).group(1)
        nodes = re.search(r"Nodes=(\S+)", block).group(1)
        print(f"⛔ {name} | {start} → {end} | Nodi: {nodes}")

if __name__ == "__main__":
    get_reservations()