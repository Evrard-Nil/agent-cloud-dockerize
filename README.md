# Automatic Containerization Tool

This tool automates the process of generating a Dockerfile and building a Docker image for a given GitHub repository. It leverages a Large Language Model (LLM) to analyze the repository's source code, identify the application's entry point, and create an appropriate Dockerfile. The tool also includes a retry mechanism to regenerate the Dockerfile based on build errors, ensuring a higher success rate for containerization.

## Features

- **Automated Dockerfile Generation**: Uses an LLM to intelligently create Dockerfiles tailored to your project's language, framework, and dependencies.
- **Repository Scanning**: Scans GitHub repositories to gather relevant code and configuration files as context for the LLM.
- **Entry Point Detection/Generation**: Automatically identifies the main entry point of your application or generates a suitable one if not explicitly found.
- **Error-Resilient Builds**: Retries Dockerfile generation and build processes, incorporating previous build errors as feedback for the LLM to refine the Dockerfile.
- **Temporary Environment**: Operates within a temporary directory to keep your local environment clean.

## Prerequisites

Before using this tool, ensure you have the following installed:

- **Git**: For cloning GitHub repositories.
- **Docker**: For building and managing Docker images.
- **Python 3.x**: The tool is written in Python.
- **An OpenAI compatible API Key**: The tool uses the OpenAI client to interact with an LLM (configured for Gemini). Set this as an environment variable: `OPENAI_API_KEY`.
- **uv**: For dependency management.

## Usage

To generate a Dockerfile and build a Docker image for a GitHub repository, run the `main.py` script with the repository URL as an argument:

```bash
uv run main.py <repository_url>
```

**Example:**

```bash
uv run main.py https://github.com/octocat/Spoon-Knife
```

The tool will:
1. Clone the specified repository into a temporary directory.
2. Scan its contents.
3. Generate a `Dockerfile` using the LLM.
4. Attempt to build a Docker image.
5. If the build fails, it will provide the error logs back to the LLM to generate a corrected `Dockerfile` and retry (up to 5 times).
6. Upon successful build, it will output the name of the generated Docker image.

## Configuration

- `MAX_RETRIES`: Maximum attempts to generate and build a Dockerfile (default: 5).
- `IMAGE_NAME_PREFIX`: Prefix for the generated Docker image names (default: `autogen-app`).
- `RELEVANT_FILE_PATTERNS`: A list of file extensions and names considered relevant for LLM context.
- `MAX_FILE_SIZE_BYTES`: Maximum size of files to read for LLM context (default: 1MB).

These can be modified directly in the `main.py` file.

## How It Works

1. **Initialization**: The script takes a GitHub repository URL as input.
2. **Repository Cloning & Scanning**: The repository is cloned into a temporary directory. The tool then walks through the cloned repository, reading the content of files matching `RELEVANT_FILE_PATTERNS` (e.g., `.py`, `package.json`, `Dockerfile`). Large files are skipped.
3. **Entry Point Identification**: An initial LLM call attempts to identify the application's main entry point (e.g., `app.py`, `index.js`) and the command to run it. If no clear entry point is found, the LLM is prompted to generate a common entry point file.
4. **Dockerfile Generation (Iterative)**:
   - The scanned code content and entry point information are sent to the LLM.
   - The LLM generates a `Dockerfile`.
   - The generated `Dockerfile` is written to the temporary repository directory.
   - A Docker build is attempted.
   - If the build fails, the build output (errors) is fed back to the LLM along with the original code context, prompting it to generate a refined `Dockerfile`. This process repeats up to `MAX_RETRIES`.
5. **Image Creation**: Upon a successful build, a Docker image is created with a unique name.


## Benchmark

```

--- Auto-Dockerizer Benchmark Report ---

Overall Performance:
  Total Repositories Benchmarked: 4
  Successful Dockerizations: 2
  Failed Dockerizations: 2
  Overall Success Rate: 50.00%

Breakdown by Tag:
  langchain:
    Successful: 0
    Failed: 2
    Success Rate: 0.00%
  no-framework:
    Successful: 2
    Failed: 0
    Success Rate: 100.00%
  notebook:
    Successful: 0
    Failed: 1
    Success Rate: 0.00%
  playwright:
    Successful: 0
    Failed: 1
    Success Rate: 0.00%
  python:
    Successful: 2
    Failed: 2
    Success Rate: 50.00%
  require-files:
    Successful: 0
    Failed: 1
    Success Rate: 0.00%

--- End Report ---
```

## Backlog

- [ ] Call specific scripts for different frameworks (e.g. https://github.com/zavodil/near-docker-runner/tree/main/runner for lanchain)
- [ ] Create PR to the repository with the generated Dockerfile
- [ ] Create the configuration file for the deployment
- [ ] Handle file requirements
