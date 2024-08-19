import os
import torch
import argparse
import logging
from datetime import datetime
from git import Repo, InvalidGitRepositoryError
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
import glob
import fnmatch
import sys
import traceback
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "THUDM/codegeex4-all-9b"
FINE_TUNED_MODEL_DIR = "./fine_tuned_models"

# Workaround for is_torch_npu_available
import transformers.utils
if not hasattr(transformers.utils, 'is_torch_npu_available'):
    transformers.utils.is_torch_npu_available = lambda: False

def setup_repo(repo_path, branch):
    """
    Sets up the Git repository at the given path, checking out the specified branch if provided.
    """
    logger.debug(f"Setting up repository: {repo_path}, branch: {branch}")
    if not os.path.exists(repo_path):
        logger.error(f"Error: The specified path {repo_path} does not exist.")
        return False

    try:
        repo = Repo(repo_path)
        if not repo.branches:
            logger.error(f"Error: The specified path {repo_path} is not a valid Git repository.")
            return False

        if branch and branch not in repo.branches:
            logger.error(f"Error: The specified branch '{branch}' does not exist in this repository.")
            return False

        if branch:
            repo.git.checkout(branch)
            logger.info(f"Switched to branch '{branch}'")
        else:
            logger.info(f"Using current branch: {repo.active_branch.name}")
        return True
    except InvalidGitRepositoryError:
        logger.error(f"Error: The specified path {repo_path} is not a valid Git repository.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in setup_repo: {e}")
        return False

def pull_github_repo(repo_path):
    """
    Pulls the latest changes from the remote GitHub repository.
    """
    logger.debug(f"Pulling latest changes for repository: {repo_path}")
    try:
        repo = Repo(repo_path)
        origin = repo.remotes.origin
        origin.pull()
        logger.info(f"Repository updated at {datetime.now()}")
    except Exception as e:
        logger.error(f"Error pulling repository: {e}")

def load_model(model_path=None, use_4bit=False, use_cpu=False):
    """
    Loads the pre-trained model and tokenizer, optionally using 4-bit quantization or CPU.
    """
    if model_path is None:
        model_path = MODEL_NAME
    logger.info(f"Attempting to load model: {model_path}")

    try:
        device = torch.device("cpu") if use_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load model with specific parameters
        if use_4bit:
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True, device_map="auto")
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map="auto" if device.type == "cuda" else None)

        if device.type == "cpu":
            model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def get_gitignore_patterns(repo_path):
    """
    Reads the .gitignore file from the repository and returns the patterns.
    """
    gitignore_path = os.path.join(repo_path, '.gitignore')
    if not os.path.exists(gitignore_path):
        logger.debug("No .gitignore file found")
        return []
    
    with open(gitignore_path, 'r') as f:
        patterns = f.read().splitlines()

    patterns = [p.strip() for p in patterns if p.strip() and not p.startswith('#')]
    logger.debug(f"Gitignore patterns: {patterns}")
    return patterns

def should_ignore(file_path, ignore_patterns):
    """
    Checks if a file should be ignored based on the provided patterns.
    """
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(file_path, pattern):
            return True
    return False

def get_project_metadata(repo_path):
    """
    Extracts project metadata, such as project name, branch, and languages used.
    """
    repo = Repo(repo_path)
    project_name = os.path.basename(repo_path)
    main_branch = repo.active_branch.name
    languages = set(os.path.splitext(f)[1][1:] for f in os.listdir(repo_path) if os.path.isfile(os.path.join(repo_path, f)))
    return f"Project: {project_name}\nMain Branch: {main_branch}\nLanguages: {', '.join(languages)}\n\n"

def prepare_dataset(repo_path, tokenizer):
    """
    Prepares the dataset for model fine-tuning from the code files in the repository.
    """
    logger.info("Preparing dataset")
    ignore_patterns = get_gitignore_patterns(repo_path)

    file_extensions = [
        '*.py', '*.js', '*.ts', '*.jsx', '*.tsx', '*.java', '*.cpp', '*.c', '*.h', '*.cs', '*.go',
        '*.rb', '*.php', '*.swift', '*.kt', '*.rs', '*.scala', '*.m', '*.mm', '*.groovy', '*.pl',
        '*.sh', '*.bash', '*.css', '*.scss', '*.less', '*.html', '*.xml', '*.json', '*.yaml', '*.yml',
        '*.md', '*.sql', '*.r', '*.ipynb', '*.dart', '*.lua', '*.jl', '*.ex', '*.exs', '*.erl', '*.hs',
        '*.fs', '*.fsx', '*.clj', '*.cljs', '*.coffee', '*.elm', '*.f', '*.f90', '*.f95', '*.zig',
        '*.v', '*.proto', '*.sol', '*.tf', '*.cmake', '*.gradle', '*.bat', '*.ps1'
    ]

    code_files = []
    file_type_count = defaultdict(int)

    for ext in file_extensions:
        for file in glob.glob(f"{repo_path}/**/{ext}", recursive=True):
            rel_path = os.path.relpath(file, repo_path)
            if not should_ignore(rel_path, ignore_patterns):
                code_files.append(file)
                file_type_count[ext[1:]] += 1  # Remove the '*.' from the extension

    logger.info(f"Total files found: {len(code_files)}")

    project_metadata = get_project_metadata(repo_path)
    code_samples = []
    for file in code_files:
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                code_samples.append(f"{project_metadata}File: {os.path.relpath(file, repo_path)}\n\n{content}")
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    dataset = Dataset.from_dict({"text": code_samples})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Report included files
    print("\nFiles included in the dataset:")
    for ext, count in sorted(file_type_count.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"{ext}: {count}")
    print(f"Total files: {sum(file_type_count.values())}")

    return tokenized_dataset

def fine_tune_model(repo_path, model, tokenizer):
    """
    Fine-tunes the model using the prepared dataset.
    """
    logger.info("Starting model fine-tuning")

    try:
        # Prepare the dataset
        train_dataset = prepare_dataset(repo_path, tokenizer)

        # Add a special token for this project
        project_token = f"<{os.path.basename(repo_path)}>"
        tokenizer.add_special_tokens({'additional_special_tokens': [project_token]})
        model.resize_token_embeddings(len(tokenizer))

        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query_key_value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Prepare the model with LoRA
        model = get_peft_model(model, lora_config)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=FINE_TUNED_MODEL_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            save_steps=10_000,
            save_total_limit=2,
            fp16=False,
            gradient_checkpointing=True,
            learning_rate=1e-4,
            warmup_steps=100,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            max_grad_norm=0.3,
            dataloader_pin_memory=False,
        )

        # Set up trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        # Start training
        trainer.train()

        # Save the fine-tuned model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(FINE_TUNED_MODEL_DIR, f"model_{timestamp}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"Fine-tuned model saved in {output_dir}")
        return output_dir
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        return None

def generate_report(report_type, model, tokenizer, repo_path):
    """
    Generates a report based on the fine-tuned model's understanding of the repository.
    """
    logger.info(f"Generating {report_type} report")
    project_metadata = get_project_metadata(repo_path)
    project_token = f"<{os.path.basename(repo_path)}>"
    prompt = f"{project_token}\n{project_metadata}\nGenerate a detailed {report_type} report for this software project. Include specific examples and recommendations where applicable."

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs, max_length=2048, num_return_sequences=1)
        report_content = tokenizer.decode(output[0], skip_special_tokens=True)
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("Out of memory error. Try reducing input size or using CPU.")
            report_content = "Error: Out of memory. Unable to generate report."
        else:
            logger.error(f"Error generating report: {e}")
            report_content = f"Error generating report: {e}"
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        report_content = f"Error generating report: {e}"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_report = f"""# {report_type.capitalize()} Report

Generated on: {timestamp}
Using model: {MODEL_NAME}

{report_content}

---
This report was automatically generated by AI based on the fine-tuned model's understanding of the repository.
"""

    filename = f"{report_type.replace(' ', '_')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w") as f:
        f.write(full_report)

    logger.info(f"{report_type.capitalize()} report generated and saved as {filename}.")

def get_available_models():
    """
    Retrieves a list of available models, including fine-tuned ones.
    """
    models = [MODEL_NAME]  # Add the base model
    if os.path.exists(FINE_TUNED_MODEL_DIR):
        fine_tuned_models = [os.path.join(FINE_TUNED_MODEL_DIR, d) for d in os.listdir(FINE_TUNED_MODEL_DIR) if os.path.isdir(os.path.join(FINE_TUNED_MODEL_DIR, d))]
        models.extend(sorted(fine_tuned_models, key=os.path.getmtime, reverse=True))
    logger.debug(f"Available models: {models}")
    return models

def select_model(models):
    """
    Displays available models and allows the user to select one.
    """
    print("\nAvailable models:")
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")
    while True:
        try:
            choice = int(input("Enter the number of the model you want to use: "))
            if 1 <= choice <= len(models):
                return models[choice-1]
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

def chat_interface(model, tokenizer, repo_path):
    """
    Launches a chat interface with the model, streaming the response token by token.
    """
    if model is None or tokenizer is None:
        logger.error("Model or tokenizer is not loaded. Exiting chat mode.")
        print("AI: Sorry, the model is not available. Please load the model first.")
        return

    logger.info("Entering chat mode")
    print("\nEntering chat mode. Type 'exit' to return to the main menu.")
    project_metadata = get_project_metadata(repo_path)
    project_token = f"<{os.path.basename(repo_path)}>"

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        try:
            # Prepare input prompt
            full_prompt = f"{project_token}\n{project_metadata}\nUser: {user_input}\nAI:"
            inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

            # Use streaming to output tokens as they are generated
            with torch.no_grad():
                streamer = transformers.TextIteratorStreamer(tokenizer, skip_special_tokens=True)
                generation_kwargs = {
                    "inputs": inputs["input_ids"],
                    "max_length": 1024,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.9,
                    "streamer": streamer
                }

                # Start generating tokens in a background thread and stream them
                generation_thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
                generation_thread.start()

                # Iterate over the generated tokens as they are streamed
                print("AI: ", end="", flush=True)
                for new_text in streamer:
                    print(new_text, end="", flush=True)

                print()  # Finish the AI response with a newline

        except Exception as e:
            logger.error(f"Error during chat interaction: {e}")
            print("AI: I'm sorry, I encountered an error while processing your request.")


def main():
    """
    Main function to handle argument parsing and execute the corresponding functionality.
    """
    parser = argparse.ArgumentParser(description="AI-powered Code Analysis Tool using CodeGeeX4")
    parser.add_argument("repo_path", help="Path to the local Git repository")
    parser.add_argument("-b", "--branch", help="Branch name (optional)", default=None)
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of CUDA")
    args = parser.parse_args()

    logger.info(f"Starting script with arguments: {args}")

    try:
        if not setup_repo(args.repo_path, args.branch):
            logger.error("Failed to set up repository. Exiting.")
            return

        models = get_available_models()
        logger.debug(f"Available models: {models}")
        current_model_path = models[0] if models else MODEL_NAME
        logger.info(f"Using model: {current_model_path}")

        model, tokenizer = load_model(current_model_path, use_4bit=args.use_4bit, use_cpu=args.use_cpu)
        if model is None or tokenizer is None:
            logger.error("Unable to load the model. Exiting.")
            return

        while True:
            print("\nOptions:")
            print("1. Pull GitHub repository")
            print("2. Fine-tune model")
            print("3. Generate security report")
            print("4. Generate bug report")
            print("5. Generate missing features report")
            print("6. Select a different model")
            print("7. Enter chat mode")
            print("8. Exit")

            choice = input("Enter your choice: ")
            logger.debug(f"User choice: {choice}")

            if choice == '1':
                pull_github_repo(args.repo_path)
            elif choice == '2':
                fine_tuned_model_path = fine_tune_model(args.repo_path, model, tokenizer)
                if fine_tuned_model_path:
                    models = get_available_models()  # Refresh the list of models
                    current_model_path = fine_tuned_model_path
                    model, tokenizer = load_model(current_model_path, use_4bit=args.use_4bit, use_cpu=args.use_cpu)
            elif choice == '3':
                generate_report("security", model, tokenizer, args.repo_path)
            elif choice == '4':
                generate_report("bug", model, tokenizer, args.repo_path)
            elif choice == '5':
                generate_report("missing features", model, tokenizer, args.repo_path)
            elif choice == '6':
                new_model_path = select_model(models)
                if new_model_path != current_model_path:
                    current_model_path = new_model_path
                    model, tokenizer = load_model(current_model_path, use_4bit=args.use_4bit, use_cpu=args.use_cpu)
            elif choice == '7':
                chat_interface(model, tokenizer, args.repo_path)
            elif choice == '8':
                logger.info("Exiting the program.")
                break
            else:
                print("Invalid choice. Please try again.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
