#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import tempfile
import time
import openai


# --- Configuration ---
MAX_RETRIES = 5  # Maximum attempts to generate and build a Dockerfile
IMAGE_NAME_PREFIX = "autogen-app"  # Prefix for the generated docker image

# List of common code file extensions and important filenames
# This helps in identifying relevant files to pass to the LLM
# We'll try to read these as text. Binary files might cause issues if not filtered.
# For simplicity, we focus on text-based files.
RELEVANT_FILE_PATTERNS = [
    # Python
    ".py",
    "requirements.txt",
    "Pipfile",
    "setup.py",
    "pyproject.toml",
    # JavaScript/TypeScript
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    "package.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "tsconfig.json",
    # Java
    ".java",
    "pom.xml",
    "build.gradle",
    "settings.gradle",
    # Go
    ".go",
    "go.mod",
    "go.sum",
    # Ruby
    ".rb",
    "Gemfile",
    "Gemfile.lock",
    # PHP
    ".php",
    "composer.json",
    "composer.lock",
    # C#
    ".cs",
    ".csproj",
    # C/C++
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    "Makefile",
    "CMakeLists.txt",
    # Web
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".less",
    # Config
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".ini",
    ".toml",
    ".env",
    # Shell
    ".sh",
    ".bash",
    # Docker (to provide as context if one already exists)
    "Dockerfile",
    ".dockerignore",
    # Rust
    ".rs",
    "Cargo.toml",
    "Cargo.lock",
    # Jupyter Notebooks
    ".ipynb",
    # Swift
    ".swift",
    "Package.swift",
    # Kotlin
    ".kt",
    ".kts",
    # Scala
    ".scala",
    ".sbt",
]

# Max file size to read (e.g., 1MB) to avoid issues with very large files
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024


# Get key from env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta",
)

dockerignore = """
# Git
.git
.gitignore

# Docker
Dockerfile
.dockerignore
docker-compose.yml
docker-compose.*.yml

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
venv
ENV
env.bak
venv.bak
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
.pytest_cache
.mypy_cache
.ruff_cache
dist
build
*.egg-info/
.ipynb_checkpoints
.pytest_cache
.vscode/
.idea/

# Node.js
node_modules
npm-debug.log
yarn-debug.log
yarn-error.log
.pnp
.pnp.js
.parcel-cache
.next
.nuxt
.vuepress/dist
.serverless
.fusebox
.temp

# Ruby
.bundle
vendor/bundle
.rake_tasks
.sass-cache
.byebug_history
.rspec
.rubocop_todo.yml
.yardoc
.gem

# Java
*.class
*.jar
*.war
*.ear
target/
build/
.gradle
.project
.classpath
.settings
bin/

# Go
*.exe
*.dll
*.so
*.dylib
*.o
*.a
*.test
vendor/

# macOS
.DS_Store

# Windows
Thumbs.db
ehthumbs.db
Desktop.ini

# Linux
*.swp

# Logs and temporary files
*.log
*.tmp
tmp/
temp/

# Editor/IDE specific files
.vscode/
.idea/
*.sublime-project
*.sublime-workspace
*.code-workspace
*.iml

# Environment variables
.env
.env.*
*.env
*.env.*

# Other common exclusions
coverage/
debug/
"""


def chat_completion(
    prompt: str, system_prompt: str = "You are a helpful AI assistant."
) -> str:
    """
    LLM call using OpenAI client with Gemini base URL.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model="gemini-2.5-flash-preview-05-20",
        messages=messages,
        temperature=0.5,
    )
    return response.choices[0].message.content


# --- Helper Functions ---
def command_exists(command: str) -> bool:
    """Check if a command exists on the system."""
    try:
        subprocess.run(
            [command, "--version"], capture_output=True, check=True, text=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_notebook_to_script(
    notebook_path: str, output_dir: str
) -> tuple[str | None, str | None]:
    """
    Converts a Jupyter notebook to a Python script using nbconvert.
    Returns (script_path, script_content) or (None, None) on failure.
    """
    print(f"Converting notebook: {notebook_path} to script in {output_dir}")
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get the base name of the notebook without extension
        notebook_name = os.path.splitext(os.path.basename(notebook_path))[0]
        script_path = os.path.join(output_dir, f"{notebook_name}.py")

        process = subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "script",
                notebook_path,
                "--output-dir",
                output_dir,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"nbconvert output:\n{process.stdout}")
        if process.stderr:
            print(f"nbconvert errors:\n{process.stderr}")

        if os.path.exists(script_path):
            with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()

            # Post-process the script content to clean up markdown and shell commands
            cleaned_lines = []
            for line in raw_content.splitlines():
                stripped_line = line.strip()
                # Remove lines that are purely markdown comments from nbconvert or shell commands
                # nbconvert typically converts markdown to lines starting with '# In[ ]:' or '# %%'
                # Shell commands start with '!'
                # Magic commands start with '%%'
                if stripped_line.startswith("# In["):
                    continue  # Skip nbconvert cell markers
                if stripped_line.startswith("!"):
                    # Comment out shell commands instead of removing them entirely,
                    # in case they are needed for context but shouldn't be executed directly.
                    cleaned_lines.append(f"# {line}")
                elif stripped_line.startswith("%%"):
                    # Comment out magic commands
                    cleaned_lines.append(f"# {line}")
                else:
                    cleaned_lines.append(line)

            content = "\n".join(cleaned_lines)

            # Write cleaned content back to the script file
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Successfully converted and cleaned script: {script_path}")
            return script_path, content
        else:
            print(f"Error: Converted script not found at {script_path}")
            return None, None
    except FileNotFoundError:
        print(
            "Error: jupyter or nbconvert command not found. Please ensure Jupyter and nbconvert are installed."
        )
        return None, None
    except subprocess.CalledProcessError as e:
        print(f"Error during nbconvert: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during notebook conversion: {e}")
        return None, None


def clone_repository(repo_url: str, target_dir: str) -> bool:
    """Clones a Git repository into the target directory."""
    print(f"Cloning repository: {repo_url} into {target_dir}")
    try:
        process = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, target_dir],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            print(f"Error cloning repository: {process.stderr}")
            return False
        print("Repository cloned successfully.")
        return True
    except FileNotFoundError:
        print(
            "Error: git command not found. Please ensure Git is installed and in your PATH."
        )
        return False
    except Exception as e:
        print(f"An unexpected error occurred during cloning: {e}")
        return False


def scan_repository_content(repo_path: str) -> str:
    """
    Scans the repository, reads relevant files, and concatenates their content.
    Skips very large files or unreadable files.
    """
    print(f"Scanning repository content at: {repo_path}")
    all_content = []
    for root, _, files in os.walk(repo_path):
        # Skip .git directory
        if ".git" in root.split(os.sep):
            continue

        for file_name in files:
            file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(file_path, repo_path)

            # Check if the file matches any of the relevant patterns
            is_relevant = False
            for pattern_or_filename in RELEVANT_FILE_PATTERNS:
                if pattern_or_filename.startswith(".") and file_name.endswith(
                    pattern_or_filename
                ):  # Check for extension
                    is_relevant = True
                    break
                elif file_name == pattern_or_filename:  # Check for exact filename match
                    is_relevant = True
                    break

            if is_relevant:
                if file_name.endswith(".ipynb"):
                    # Convert notebook to script and add its content
                    script_output_dir = os.path.join(repo_path, ".converted_scripts")
                    converted_script_path, converted_script_content = (
                        convert_notebook_to_script(file_path, script_output_dir)
                    )
                    if converted_script_path and converted_script_content:
                        relative_script_path = os.path.relpath(
                            converted_script_path, repo_path
                        )
                        all_content.append(
                            f"\n--- File: {relative_script_path} (Converted from {relative_path}) ---\n{converted_script_content}"
                        )
                        print(f"Added converted script content: {relative_script_path}")
                    else:
                        all_content.append(
                            f"\n--- File: {relative_path} (Notebook conversion failed) ---\n"
                        )
                        print(f"Notebook conversion failed for: {relative_path}")
                else:
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size > MAX_FILE_SIZE_BYTES:
                            print(
                                f"Skipping large file: {relative_path} (size: {file_size} bytes)"
                            )
                            all_content.append(
                                f"\n--- File: {relative_path} (Skipped: Too large) ---\n"
                            )
                            continue

                        with open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            content = f.read()
                        all_content.append(
                            f"\n--- File: {relative_path} ---\n{content}"
                        )
                        print(f"Read file: {relative_path}")
                    except Exception as e:
                        all_content.append(
                            f"\n--- File: {relative_path} (Error reading: {e}) ---\n"
                        )
                        print(f"Error reading file {relative_path}: {e}")

    if not all_content:
        print("Warning: No relevant files found or read from the repository.")
        return "No relevant file content found in the repository."

    return "\n".join(all_content)


def build_docker_image(context_path: str, image_name: str) -> tuple[bool, str]:
    """
    Attempts to build a Docker image using the Dockerfile in the context_path.
    Returns (success_status, build_output).
    """
    print(
        f"Attempting to build Docker image: {image_name} from context: {context_path}"
    )
    dockerfile_path = os.path.join(context_path, "Dockerfile")
    if not os.path.exists(dockerfile_path):
        return False, "Dockerfile not found in the context path."

    try:
        process = subprocess.Popen(
            ["docker", "build", "-t", image_name, "."],
            cwd=context_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,  # Decode output as text
            bufsize=1,  # Line-buffered
        )

        output_lines = []
        # print("--- Docker Build Output (Streaming) ---")
        if process.stdout:  # Ensure stdout is not None before iterating
            for line in process.stdout:
                # print(line, end="")  # Print in real-time
                output_lines.append(line)
        # print("---------------------------------------")

        process.wait()  # Wait for the process to complete
        output = "".join(output_lines)

        if process.returncode == 0:
            print(f"Docker image '{image_name}' built successfully.")
            return True, output
        else:
            print(f"Docker build failed for image '{image_name}'.")
            print("--- Full Docker Build Output ---")
            print(output)
            return False, output
    except FileNotFoundError:
        error_msg = "Error: docker command not found. Please ensure Docker is installed and running."
        print(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during Docker build: {e}"
        print(error_msg)
        return False, error_msg


def run_and_check_image(image_name: str) -> tuple[bool, str]:
    """
    Runs the Docker image without configuration and checks its output using LLM.
    Returns (True, LLM_analysis) if output is OK or fails due to config,
    (False, LLM_analysis) if it fails due to missing dependencies.
    """
    print(f"Running Docker image '{image_name}' for final check...")
    try:
        # Run the container, capture output, and remove it after exit
        process = subprocess.run(
            ["docker", "run", "--rm", image_name],
            capture_output=True,
            text=True,
            check=False,  # Do not raise an exception for non-zero exit codes
            timeout=60,  # Add a timeout to prevent hanging containers
        )
        container_output = process.stdout + "\n" + process.stderr
        print(f"Container exited with code: {process.returncode}")
        print("--- Container Output ---")
        print(container_output)
        print("------------------------")

        # Use LLM to analyze the output
        analysis_prompt = f"""
        Analyze the following output from a Docker container.
        The container was run without any specific configuration (environment variables, volume mounts).
        Determine if the application:
        1. Ran successfully and produced expected output (indicate 'STATUS: OK').
        2. Failed due to missing *configuration* (e.g., missing API keys, database connection strings, specific file paths).
           If so, explain why it's a configuration issue (indicate 'STATUS: CONFIG_ERROR').
        3. Failed due to missing *dependencies* (e.g., Python packages not found, Node.js modules not installed, Java classes not found, missing shared libraries).
           If so, explain why it's a dependency issue (indicate 'STATUS: DEPENDENCY_ERROR').

        Provide a concise analysis and clearly state the STATUS.

        Container Output:
        ```
        {container_output}
        ```
        """
        llm_analysis = chat_completion(
            analysis_prompt,
            system_prompt="You are an expert AI assistant that analyzes Docker container logs to identify the root cause of application failures, distinguishing between configuration and dependency issues.",
        )
        print(f"LLM Analysis of Container Output: {llm_analysis}")

        if "STATUS: DEPENDENCY_ERROR" in llm_analysis:
            return False, llm_analysis
        else:
            return True, llm_analysis

    except FileNotFoundError:
        error_msg = "Error: docker command not found. Please ensure Docker is installed and running."
        print(error_msg)
        return (
            False,
            error_msg,
        )  # Treat as a failure to proceed, though not a dependency error from the app itself
    except subprocess.TimeoutExpired:
        error_msg = "Error: Docker container timed out after 60 seconds. This might indicate a hanging process or a very long startup."
        print(error_msg)
        # If it times out, it's hard to say if it's dependency or config.
        # For now, let's treat it as a failure that needs regeneration.
        return False, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during Docker run check: {e}"
        print(error_msg)
        return False, error_msg


def find_or_create_entrypoint(scanned_code_content: str) -> str:
    """
    Finds the entry point of the application from the scanned code content.
    If no entry point is found, it generates code for a common entry point.
    """
    print("Attempting to find or create application entry point...")

    # Prompt to find the entry point
    find_entrypoint_prompt = f"""
    Analyze the following project source code and file structure to identify the main entry point of the application.
    The entry point could be a main function, a script that starts a web server, or a primary executable file.
    If you find a clear entry point, provide its relative path and the command to run it (e.g., 'python app.py', 'npm start', 'java -jar app.jar').
    If no clear entry point is found, state "NO_ENTRY_POINT_FOUND".

    Project source code and file structure:
    ```
    {scanned_code_content}
    ```

    Example of a valid response for a Python app:
    Path: app.py
    Command: python app.py

    Example of a valid response for a Node.js app:
    Path: server.js
    Command: node server.js

    Example of a response if no entry point is found:
    NO_ENTRY_POINT_FOUND
    """
    entrypoint_response = chat_completion(
        find_entrypoint_prompt,
        system_prompt="You are an expert AI assistant that identifies application entry points.",
    )
    print(f"LLM Entry Point Response: {entrypoint_response}")

    if "NO_ENTRY_POINT_FOUND" in entrypoint_response:
        print("No clear entry point found. Generating a default entry point...")

        is_notebook_content = any(
            ".ipynb" in line
            for line in scanned_code_content.split("\n")
            if line.startswith("--- File:")
        )

        if is_notebook_content:
            print("Detected notebook content. Requesting equivalent script.")
            generate_entrypoint_prompt = f"""
            The following project source code includes content from a Jupyter notebook.
            Please generate an equivalent runnable script (e.g., a Python script if the notebook is Python-based)
            that can serve as the main entry point for the application.
            The script should contain the necessary code to execute the core logic found in the notebook.
            Provide only the code for the entry point file, without any additional text or explanations.
            Indicate the suggested filename at the beginning of your response, like "Filename: script.py".

            Project source code and file structure:
            ```
            {scanned_code_content}
            ```
            """
        else:
            generate_entrypoint_prompt = f"""
            Based on the following project source code, generate a simple, common entry point file (e.g., app.py for Python, index.js for Node.js)
            that could serve as the main starting point for the application.
            Assume the application is a simple web service or script.
            Provide only the code for the entry point file, without any additional text or explanations.
            Indicate the suggested filename at the beginning of your response, like "Filename: app.py".

            Project source code and file structure:
            ```
            {scanned_code_content}
            ```
            """

        generated_code_response = chat_completion(
            generate_entrypoint_prompt,
            system_prompt="You are an expert AI assistant that generates application entry points and converts notebooks to scripts.",
        )
        print(f"LLM Generated Entry Point Code Response: {generated_code_response}")

        # Clean backticks and ensure the response is properly formatted
        generated_code_response = generated_code_response.replace("```", "").strip()
        generated_code_response = generated_code_response.replace(">>>", "").strip()
        # Extract filename and content
        lines = generated_code_response.strip().split("\n")
        filename_line = lines[0]
        if filename_line.lower().startswith("filename:"):
            filename = filename_line.split(":", 1)[1].strip()
            content = "\n".join(lines[1:]).strip()
            return f"Generated Entry Point:\nFilename: {filename}\nContent:\n{content}"
        else:
            print(
                "Could not parse generated entry point filename. Returning raw response."
            )
            return generated_code_response
    else:
        print("Entry point identified.")
        return entrypoint_response


def generate_dependency_install_instructions(
    scanned_code_content: str, entry_point_info: str
) -> str:
    """
    Generates Dockerfile instructions for installing project dependencies
    based on scanned code content and entry point information.
    """
    print("Generating Dockerfile dependency installation instructions...")
    dependency_prompt = f"""
    Analyze the following project source code and entry point information.
    Based on the detected programming language, framework, and dependency files (e.g., requirements.txt, package.json, pom.xml, go.mod, Gemfile, composer.json),
    generate *only* the Dockerfile instructions required to install these dependencies.
    If no specific dependency installation is needed or detectable, return an empty string or a comment indicating so.
    If the dependency installation requires specific commands or tools, include those in the instructions. For example if cmake is needed, include the command to install it.

    Project source code and file structure:
    ```
    {scanned_code_content}
    ```

    Entry point information:
    ```
    {entry_point_info}
    ```

    Provide only the Dockerfile instructions.
    """
    instructions_response = chat_completion(
        dependency_prompt,
        system_prompt="You are an expert AI assistant that generates precise Dockerfile instructions for dependency installation.",
    )
    print(f"LLM Dependency Install Instructions Response:\n{instructions_response}")
    # Clean backticks and ensure the response is properly formatted
    instructions_response = instructions_response.replace("```dockerfile", "").strip()
    instructions_response = instructions_response.replace("```", "").strip()
    instructions_response = instructions_response.replace(">>> dockerfile", "").strip()
    instructions_response = instructions_response.replace(">>>", "").strip()
    return instructions_response


def gather_environment_variables_info(scanned_code_content: str) -> str:
    """
    Gathers information about potential environment variables used by the application.
    """
    print("Gathering environment variable information...")
    env_var_prompt = f"""
    Analyze the following project source code and identify any potential environment variables that the application might use.
    Look for common patterns like `os.environ.get()`, `process.env.VAR_NAME`, configuration files (.env, config.json, etc.) that might reference environment variables, or any other indicators of external configuration.
    List the names of the environment variables you find or infer, and briefly explain their likely purpose if discernible.
    If no environment variables are explicitly found or strongly inferred, state "No explicit environment variables found or inferred."

    Project source code and file structure:
    ```
    {scanned_code_content}
    ```

    Provide the information in a clear, concise list without duplicates.
    """
    env_var_info = chat_completion(
        env_var_prompt,
        system_prompt="You are an expert AI assistant that identifies and explains environment variables used in software projects.",
    )
    print(f"LLM Environment Variable Info:\n{env_var_info}")
    return env_var_info


# --- Main Script Logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Generate a Dockerfile for a GitHub repository using LLM."
    )
    parser.add_argument("repo_url", help="URL of the GitHub repository to process.")
    args = parser.parse_args()

    # Check for prerequisites
    if not command_exists("git"):
        print("Error: Git is not installed or not found in PATH. Please install Git.")
        return
    if not command_exists("docker"):
        print(
            "Error: Docker is not installed, not found in PATH, or the Docker daemon is not running. Please install and start Docker."
        )
        return

    repo_url = args.repo_url
    last_error_output = ""

    # Generate a unique image name with a timestamp
    timestamp = int(time.time())
    current_image_name = f"{IMAGE_NAME_PREFIX}-{timestamp}:latest"

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        if not clone_repository(repo_url, temp_dir):
            return

        scanned_code_content = scan_repository_content(temp_dir)
        if (
            "No relevant file content found" in scanned_code_content
            and len(scanned_code_content) < 100
        ):
            print("Exiting due to lack of scannable content.")
            return

        entry_point_info = find_or_create_entrypoint(scanned_code_content)
        print(f"Entry point information: {entry_point_info}")

        dependency_install_instructions = generate_dependency_install_instructions(
            scanned_code_content, entry_point_info
        )

        environment_variables_info = gather_environment_variables_info(
            scanned_code_content
        )
        print(f"Environment Variables Info: {environment_variables_info}")

        generate_dockerfile(
            last_error_output,
            current_image_name,
            temp_dir,
            scanned_code_content,
            entry_point_info,
            dependency_install_instructions,
            environment_variables_info,
        )

    print("\nScript finished.")


def generate_dockerfile(
    last_error_output,
    current_image_name,
    temp_dir,
    scanned_code_content,
    entry_point_info: str,
    dependency_install_instructions: str,
    environment_variables_info: str,
):
    build_success = False
    system_prompt = """You are an expert AI assistant that specializes in generating correct and efficient Dockerfiles for various software projects.
    Analyze the provided source code and any previous build errors to create a complete and runnable Dockerfile. 
    Only output the content of the Dockerfile, without any additional text or explanations. 
    Do not include prefixes like '>>> dockerfile' or '```dockerfile'. 
    Do not indent the Dockerfile content. 
    Ensure the Dockerfile is suitable for the detected programming language and framework, and includes all necessary dependencies and configurations. 
    Do not include environment variables in the Dockerfile unless explicitly requested.
    All environment variables will be provided at runtime and should not be hardcoded in the Dockerfile.
    If you encounter errors, analyze them and adjust the Dockerfile accordingly."""

    prompt_parts = []

    prompt_parts.append(
        "Please generate a Dockerfile for an application with the following source code and file structure."
    )
    prompt_parts.append(
        "Analyze the files to determine the language, framework, dependencies, and appropriate entry point."
    )

    prompt_parts.append(
        f"\nCollected code and configuration files:\n```\n{scanned_code_content}\n```"
    )
    prompt_parts.append(f"\nEntry point information:\n```\n{entry_point_info}\n```")
    if dependency_install_instructions:
        prompt_parts.append(
            f"\nDependency Installation Instructions:\n```dockerfile\n{dependency_install_instructions}\n```\n"
        )
        prompt_parts.append(
            "Integrate these instructions into the Dockerfile at the appropriate stage."
        )
    else:
        prompt_parts.append(
            "\nNo specific dependency installation instructions were generated, infer dependencies from code."
        )

    # Add environment variables info to the prompt
    if environment_variables_info:
        prompt_parts.append(
            f"\nEnvironment Variable Information:\n```\n{environment_variables_info}\n```\n"
        )
        prompt_parts.append(
            "Consider these environment variables for Dockerfile configuration (e.g., ENV instructions, default values)."
        )
    else:
        prompt_parts.append(
            "\nNo explicit environment variables were found or inferred."
        )

    prompt_parts.append("\nGenerate only the Dockerfile content.")

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n--- Attempt {attempt} of {MAX_RETRIES} ---")
        prompt_parts.append(
            f"Attempt {attempt} of {MAX_RETRIES} to generate a Dockerfile for the application."
        )

        if last_error_output:
            prompt_parts.append("The previous Dockerfile build attempt failed.")
            prompt_parts.append(
                "Please analyze the following error and the project code to generate a corrected Dockerfile."
            )
            prompt_parts.append(f"Error log:\n```\n{last_error_output}\n```")

        current_prompt = "\n\n".join(prompt_parts)

        dockerfile_content = chat_completion(current_prompt, system_prompt)
        # remove ticks
        dockerfile_content = dockerfile_content.replace("```dockerfile", "").strip()
        dockerfile_content = dockerfile_content.replace("```", "").strip()
        dockerfile_content = dockerfile_content.replace(">>> dockerfile", "").strip()
        dockerfile_content = dockerfile_content.replace(">>>", "").strip()

        if not dockerfile_content or dockerfile_content.strip() == "":
            print(
                "LLM returned empty Dockerfile content. Cannot proceed with this attempt."
            )
            last_error_output = "LLM returned empty content."
            if attempt < MAX_RETRIES:
                print("Retrying...")
                time.sleep(2)
                continue
            else:
                print("Max retries reached. LLM failed to provide content.")
                break

        dockerfile_path = os.path.join(temp_dir, "Dockerfile")
        dockerignore_path = os.path.join(temp_dir, ".dockerignore")
        try:
            with open(dockerfile_path, "w", encoding="utf-8") as f:
                f.write(dockerfile_content)
            with open(dockerignore_path, "w", encoding="utf-8") as f:
                f.write(dockerignore.strip())
            print(f"Generated Dockerfile written to: {dockerfile_path}")
            print("--- Generated Dockerfile Content ---")
            print(dockerfile_content)
            print("------------------------------------")
        except Exception as e:
            print(f"Error writing Dockerfile: {e}")
            last_error_output = f"Error writing Dockerfile: {e}"
            if attempt < MAX_RETRIES:
                print("Retrying...")
                continue
            else:
                print("Max retries reached. Failed to write Dockerfile.")
                break

        build_success, build_output = build_docker_image(temp_dir, current_image_name)

        if build_success:
            print(f"\nDockerfile built successfully on attempt {attempt}!")
            print(f"Image '{current_image_name}' is ready.")

            # --- New: Final check by running the image without configuration ---
            print("\n--- Performing final image run check ---")
            run_check_success, run_check_analysis = run_and_check_image(
                current_image_name
            )

            if run_check_success:
                print("\nFinal image run check passed.")
                # print(f"LLM Analysis: {run_check_analysis}")
                break  # Dockerfile and image are good
            else:
                print(
                    "\nFinal image run check failed due to potential dependency issues."
                )
                print(f"LLM Analysis: {run_check_analysis}")
                build_success = False  # Mark as failed to trigger regeneration
                last_error_output = f"Image run check failed with potential dependency issues. LLM analysis:\n{run_check_analysis}"
                if attempt == MAX_RETRIES:
                    print(
                        "Maximum retries reached. Could not generate a working Dockerfile after run check failures."
                    )
                else:
                    print(
                        "Will attempt to regenerate Dockerfile based on run check error."
                    )
                    time.sleep(2)  # Small delay before next LLM call
        else:
            print(f"Build failed on attempt {attempt}.")
            last_error_output = build_output
            if attempt == MAX_RETRIES:
                print("Maximum retries reached. Could not build a working Dockerfile.")
            else:
                print("Will attempt to regenerate Dockerfile based on error.")
                time.sleep(2)

    if not build_success:
        print("\nFailed to generate a working Dockerfile after all attempts.")
    else:
        print(
            f"\nSuccessfully generated a Dockerfile and built the image '{current_image_name}'."
        )
        print("You can now run the image with:")
        print(f"docker run --rm {current_image_name}")


if __name__ == "__main__":
    main()
