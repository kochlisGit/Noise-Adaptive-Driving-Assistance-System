import subprocess
import time
import os

# Set the path to the Carla server executable
CARLA_SERVER_PATH = '../CarlaUE4.exe'

# Set the arguments for launching the Carla server
CARLA_SERVER_ARGS = [
    '-quality-level=Low',
    '-carla-server',
    '-RenderOffscreen',
    '-ResX=800',
    '-ResY=600'
    # '-carla-host=155.207.113.68'
]

# The name of the process to search for and kill
process_name = "Carla"

# Get a list of all running processes
output = os.popen('ps -A').read()

# Infinite loop to launch and relaunch the Carla server
while True:
    print("Launching Carla server...")

    # Launch the Carla server process
    carla_server_process = subprocess.Popen([CARLA_SERVER_PATH] + CARLA_SERVER_ARGS)

    # Wait for a second to give the process time to start
    time.sleep(5)

    # Check whether the process is still alive
    while carla_server_process.poll() is None:
        # If the process is still alive, wait for a second and check again
        time.sleep(2)

    # If the process has exited, print an error message and relaunch the server
    print("Carla server has crashed! Relaunching...")
