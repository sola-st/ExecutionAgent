import docker
from docker.errors import ImageNotFound
import os
import subprocess
import re
import time

from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage

ACTIVE_SCREEN = {
    "name": "my_screen_session",
    "id": None,
    "default_process_list": None,
    "prep_end": False
}

def ask_llm(query, system_message, model="gpt-4.1-mini"):
    with open("openai_token.txt") as opt:
        token = opt.read()
    chat = ChatOpenAI(openai_api_key=token, model=model)

    messages = [
        SystemMessage(
            content= system_message
                    ),
        HumanMessage(
            content=query
            )  
    ]
    #response_format={ "type": "json_object" }
    response = chat.invoke(messages)

    return response.content

import xml.etree.ElementTree as ET
import yaml

def xml_to_dict(element):
    """ Recursively converts XML elements to a dictionary. """
    if len(element) == 0:
        return element.text
    return {
        element.tag: {
            child.tag: xml_to_dict(child) for child in element
        }
    }

def convert_xml_to_yaml(xml_content):
    """ Converts XML content (as a string) to a YAML string. """
    # Parse the XML content from the string
    root = ET.fromstring(xml_content)
    
    # Convert XML to a dictionary
    xml_dict = xml_to_dict(root)
    
    # Convert the dictionary to a YAML string
    yaml_str = yaml.dump(xml_dict, default_flow_style=False)
    
    return yaml_str

def send_command_to_shell(container, command):
    try:
        # Send a command to the shell session
        exec_result = container.exec_run(f"bash -c '{command}'")
        
        output = exec_result.output.decode('utf-8')
        print(f"Command output:\n{output}")
        return output
    
    except Exception as e:
        return f"An error occurred while sending the command: {e}"

def get_screen_process_list(container, screen_id):
    command = "pstree -p {}".format(screen_id)
    output = execute_command_in_container_screen(container, command)
    return output

def create_screen_session(container):
    command = "apt update && apt install -y screen"
    execute_command_in_container_screen(container, command)

    command = "apt install psmisc"
    execute_command_in_container_screen(container, command)

    command = "touch /tmp/cmd_result"
    execute_command_in_container_screen(container, command)

    command = "screen -dmS my_screen_session"
    execute_command_in_container_screen(container, command)

    command = "screen -ls"
    output = execute_command_in_container_screen(container, command)
    
    
    session_id = parse_screen_sesssion_id(output)
     
    ACTIVE_SCREEN["id"] = session_id
    ACTIVE_SCREEN["default_process_list"] = get_screen_process_list(container, session_id)
    ACTIVE_SCREEN["prep_end"] = True

    command = "TZ=Europe/Berlin && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone"
    output = execute_command_in_container_screen(container, command)

def parse_screen_sesssion_id(screen_ls):
    lines = screen_ls.splitlines()
    
    for line in lines:
        if ".my_screen_session" in line:
            wanted_line = line
            break
    else:
        raise ValueError("ERROR: This is not possible, my_screen_session should be there")

    line_parts = wanted_line.split()
    for part in line_parts:
        if ".my_screen_session" in part:
            wanted_part = part
            break
    else:
        raise ValueError("ERROR 2: This is not possible, my_screen_session should be there")

    return wanted_part.split(".")[0]


def remove_duplicate_consecutive_lines(text):
    lines = text.split('\n')  # Split the text into individual lines
    result_lines = []         # List to store the unique lines
    previous_line = None       # Keep track of the last processed line

    for line in lines:
        if line != previous_line:  # Only append the line if it's different from the last one
            result_lines.append(line)
        previous_line = line       # Update the last processed line
    
    return '\n'.join(result_lines)  # Join the unique lines back into a single text block

def remove_progress_bars(text):
    try:
        with open("prompt_files/remove_progress_bars") as rpb:
            system_prompt= rpb.read()
        summary = ""
        for i in range(int(len(text)/100000)+1):
            query= "Here is the output of a command that you should clean:\n"+ text[i*100000: (i+1)*100000]
            summary += "\n" + ask_llm(query, system_prompt)
            print("CLEANED 100K CHARACTERS.........")
            print("LEN CLEANED:", len(summary))
    except Exception as e:
        print("ERRRRRROOOOOOOOOOOR IN PROGRESSSSSSSSSS:", e)

    return summary

def remove_ansi_escape_sequences(text):
    """
    Removes ANSI escape sequences from a given string.
    
    Parameters:
    text (str): The string containing ANSI escape sequences.
    
    Returns:
    str: The cleaned string without ANSI escape sequences.
    """
    # Regular expression to match ANSI escape sequences
    ansi_escape = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]')
    
    # Removing ANSI escape sequences
    clean_text = ansi_escape.sub('', text)
    
    return clean_text

def check_image_exists(image_name):
    client = docker.from_env()
    try:
        client.images.get(image_name)
        print(f"Image '{image_name}' exists.")
        return True
    except ImageNotFound:
        print(f"Image '{image_name}' does not exist.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def textify_output(output):
    # Decode bytes to string
    output_str = output

    # Regular expression pattern to match ANSI escape sequences
    ansi_escape = re.compile(r'\x1b\[([0-9;]*[A-Za-z])')

    # Remove ANSI escape sequences
    clean_output = ansi_escape.sub('', output_str)

    # Remove extra whitespace characters like \r and \n
    clean_output = clean_output
    return clean_output

def extract_test_sections(maven_output):
    # Regular expressions to match the start and end of test sections
    test_section_start = re.compile(r'Tests run: \d+, Failures: \d+, Errors: \d+, Skipped: \d+')
    test_section_end = re.compile(r'\[INFO\] .*')

    # Find all the indices where the test sections start and end
    starts = [match.start() for match in test_section_start.finditer(maven_output)]
    ends = [match.start() for match in test_section_end.finditer(maven_output)]

    # Ensure each start has a corresponding end
    sections = []
    for start in starts:
        end = next((e for e in ends if e > start), None)
        if end:
            sections.append(maven_output[start:end])
    
    # If no test sections are detected, return the original output
    if not sections:
        return maven_output

    # Join all extracted sections into a single string
    return "\n".join(sections)

def build_image(dockerfile_path, tag):
    client = docker.from_env()
    try:
        log_text = ""
        print(f"Building Docker image from {dockerfile_path} with tag {tag}...")
        image, logs = client.images.build(path=dockerfile_path, tag=tag, rm=True, nocache=True)
        for log in logs:
            if 'stream' in log:
                log_text += log['stream'].strip()
        return "Docker image built successfully.\n"
    except Exception as e:
        return f"An error occurred while building the Docker image: {e}"
        return None
import docker

def start_container(image_tag):
    client = docker.from_env()
    try:
        print(f"Running container from image {image_tag}...")
        container = client.containers.run(image_tag, command=["tail", "-f", "/dev/null"], detach=True, tty=True)
        print(f"Container {container.short_id} is running.")
        print("CREATING SCREEN SESSION")
        create_screen_session(container)
        execute_command_in_container(container, "screen -S my_screen_session -X stuff 'apt update && apt install -y coreutils\\n'")
        return container
    except Exception as e:
        print(f"ERRRRRRRRRRRR: An error occurred while running the container: {e}")
        return None

def execute_command_in_container_old(container, command):
    try:
        print(f"Executing command '{command}' in container {container.short_id}...")
        exec_result = container.exec_run(command, tty=True)
        print(f"Command output:\n{exec_result.output.decode('utf-8')}")
        clean_output = remove_progress_bars(textify_output(exec_result.output.decode('utf-8')))
        test_sections = extract_test_sections(clean_output)
        return test_sections
    except Exception as e:
        return f"An error occurred while executing the command: {e}"
        return None

def execute_command_in_container(container, command):
    try:
        # Wrap the command in a shell execution context
        shell_command = "/bin/bash -c \"{}\" > /tmp/cmd_result\n".format(command)
        #print(f"Executing command '{command}' in container {container.short_id}...")

        # Execute the command without a TTY, but with streaming output
        exec_result = container.exec_run(shell_command, tty=False)

        # Decode and process the output
        output = exec_result.output.decode('utf-8')
        #print(f"Command output:\n{output}")
        
        THRESH = 300
        WAIT = 1
        command_threshold = THRESH
        old_command_output = read_file_from_container(container, "/tmp/cmd_result")

        while get_screen_process_list(container, ACTIVE_SCREEN["id"]) != ACTIVE_SCREEN["default_process_list"]:
            print("WAITING FOR PROCESS TO FINISH...")
            print(ACTIVE_SCREEN["default_process_list"])
            print(get_screen_process_list(container, ACTIVE_SCREEN["id"]))
            time.sleep(WAIT)
            new_command_output =  read_file_from_container(container, "/tmp/cmd_result")

            if new_command_output == old_command_output:
                command_threshold -= WAIT
            else:
                command_threshold = THRESH
            
            if command_threshold <= 0:
                with open("prompt_files/command_stuck") as cst:
                    stuck_m = cst.read()
                return "The command you executed seems to take some time to finish.. Here is the output that the command has so far (it did not change for the last {} seconds):\n".format(THRESH) + old_command_output + "\n\n" + stuck_m
        return output

    except Exception as e:
        return f"An error occurred while executing the command: {e}"

def execute_command_in_container_screen(container, command):
    try:
        # Wrap the command in a shell execution context
        shell_command = "/bin/bash -c \"{}\"".format(command)
        #print(f"Executing command '{command}' in container {container.short_id}...")

        # Execute the command without a TTY, but with streaming output
        exec_result = container.exec_run(shell_command, tty=False)

        # Decode and process the output
        output = exec_result.output.decode('utf-8')
        #print(f"Command output:\n{output}")
        return output

    except Exception as e:
        return f"An error occurred while executing the command: {e}"

# Example usage:
# Start a container
#container = start_container('your_image_tag')
def stop_and_remove(container):
    container.stop()
    container.remove()
    return "Container stopped and removed successfully"
    
def run_container(image_tag, script_path):
    client = docker.from_env()
    try:
        print(f"Running container from image {image_tag}...")
        container = client.containers.run(image_tag, detach=True, tty=True)
        print(f"Container {container.short_id} is running.")
        
        # Use docker cp to copy the script into the cloned repository folder inside the container
        script_name = os.path.basename(script_path)
        container_id = container.short_id
        subprocess.run(['docker', 'cp', script_path, f'{container_id}:/app/code2flow/{script_name}'])
        print(f"Copied {script_name} to /app/code2flow/ in the container.")

        # Execute the script inside the container
        exec_result = container.exec_run(f"sh /app/code2flow/{script_name}", stderr=True, stdout=True)
        stdout = exec_result.output.decode()
        exit_code = exec_result.exit_code
        print(f"Script executed with exit code {exit_code}. Output:")
        print(stdout)
        
        return exit_code, stdout
    except Exception as e:
        print(f"An error occurred while running the container: {e}")
        return None, None
    finally:
        container.remove(force=True)
        print(f"Container {container.short_id} has been removed.")

import tarfile
import io

def create_file_tar(file_path, file_content):
    data = io.BytesIO()
    with tarfile.TarFile(fileobj=data, mode='w') as tar:
        tarinfo = tarfile.TarInfo(name=file_path)
        tarinfo.size = len(file_content)
        tar.addfile(tarinfo, io.BytesIO(file_content.encode('utf-8')))
    data.seek(0)
    return data

def write_string_to_file(container, file_content, file_path):
    try:
        # Create a tarball with the file
        tar_data = create_file_tar(file_path, file_content)

        # Copy the tarball into the container
        container.put_archive('/', tar_data)

        # Verify the file was written
        exit_code, output = container.exec_run(f"cat {file_path}")
        if exit_code == 0:
            print(f"File content in container: {output.decode('utf-8')}", file_path)
        else:
            print(f"Failed to verify the file in the container: {output.decode('utf-8')}")
    finally:
        # Stop and remove the container
        pass

def read_file_from_container(container, file_path):
    """
    Reads the content of a file within a Docker container and returns it as a string.

    Args:
    - container: The Docker container instance.
    - file_path: The path to the file inside the container.

    Returns:
    - The content of the file as a string.
    """
    # Construct the command to read the file content
    command = f'cat {file_path}'

    # Execute the command within the container
    exit_code, output = container.exec_run(cmd=command, tty=True)
    
    if exit_code == 0:
        if file_path.lower().endswith("xml"):
            return convert_xml_to_yaml(output.decode('utf-8'))
        return output.decode('utf-8')
    else:
        return f'Failed to read {file_path} in the container. Output: {output.decode("utf-8")}'

SCREEN_SESSION = ACTIVE_SCREEN["name"]
LOG_DIR        = "/tmp"
THRESH         = 300   # seconds of no change before "stuck"
WAIT           = 1     # polling interval in seconds

import uuid
import time
from docker.models.containers import Container

from .docker_helpers_static import (
    ACTIVE_SCREEN,
    get_screen_process_list,
    read_file_from_container,
    textify_output,
    remove_progress_bars,
)

def exec_in_screen_and_get_log(container: Container, cmd: str) -> tuple[int, str, str, bool]:
    """
    Improved: waits for first output before ever checking the process tree,
    then breaks only after seeing both: 1) some output, and 2) the tree back to default.
    """
    run_id   = uuid.uuid4().hex
    logfile  = f"{LOG_DIR}/{SCREEN_SESSION}_{run_id}.log"
        
    # start per‐command logging
    container.exec_run(f"screen -S {SCREEN_SESSION} -X logfile {logfile}")
    container.exec_run(f"screen -S {SCREEN_SESSION} -X log on")
    time.sleep(0.5) 

    if cmd in ['exec "$SHELL" -l', "exec '$SHELL' -l", 'exec "$SHELL" -l ', "exec '$SHELL' -l "]:
        container.exec_run(f"screen -S {SCREEN_SESSION} -X stuff 'exec /bin/bash -l\\n'", tty=False)
        special_output = read_file_from_container(container, logfile)
        container.exec_run(f"screen -S {SCREEN_SESSION} -X log off")
        return 0, f"The shell has been renewed. Here is what appears on the new terminal: {special_output}", f"The shell has been renewed. Here is what appears on the new terminal: {special_output}", False
    # prime old_output
    try:
        old_output = read_file_from_container(container, logfile)
    except Exception:
        old_output = ""

    # send the actual shell command
    container.exec_run(f"screen -S {SCREEN_SESSION} -X stuff '{cmd}\\n'", tty=False)

    stuck      = False
    threshold  = THRESH
    seen_any   = False
    grace_deadline = None   # timestamp by which we must confirm the tree

    while True:
        try:
            new_output = read_file_from_container(container, logfile)
        except Exception as e:
            new_output = f"An Error happened during executing the command:{e}"

        if new_output != old_output:
            seen_any     = True
            old_output   = new_output
            threshold    = THRESH
            grace_deadline = time.time() + 2   # give 50 ms for “tree check”
        else:
            threshold -= WAIT

        if seen_any:
            tree = get_screen_process_list(container, ACTIVE_SCREEN["id"])
            if tree == ACTIVE_SCREEN["default_process_list"]:
                time.sleep(2)
                break


        if threshold <= 0:
            stuck = True
            break

        time.sleep(WAIT)

    # stop logging
    container.exec_run(f"screen -S {SCREEN_SESSION} -X log off")
    time.sleep(2)
    old_output = read_file_from_container(container, logfile)
    # build the return values
    if stuck:
        with open("prompt_files/command_stuck") as f:
            stuck_prompt = f.read()
        cleaned = (
            "The command you executed seems to take some time to finish...\n\n"
            f"Partial output (no change for {THRESH}s):\n{ textify_output(old_output) }\n\n"
            "You can call the linux_terminal again with one of the following options:\n"
            " WAIT which would allow you to wait more for the process to finish if it makes sense based on the partial progress so far.\n"
            " TERMINATE to kill the command if necessary.\n"
            " WRITE:<your text> to send input to a command that is requiring input (some inputs such as [ENTER] might require usage of special characters to represent [ENETER] as a string, e.g, represented as a backslash n or a baskslash r).\n\n"
            + stuck_prompt
        )
        return 1, cleaned, logfile, True

    # normal completion
    clean = textify_output(old_output)
    if len(clean) > 2000:
        clean = remove_progress_bars(clean)
    return 0, clean, logfile, False


if __name__ == "__main__":
    screen_text = """There is a screen on:
        37.my_screen_session    (09/13/24 10:12:26)     (Detached)
1 Socket in /run/screen/S-root."""

    print(parse_screen_sesssion_id(screen_text))
