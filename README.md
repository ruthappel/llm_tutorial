# LLM Tutorial

A guide to working with large language models using Google Colab, APIs, Hugging Face, and virtual machines (VMs).

Note that this tutorial was first drafted in June 2023 for a research project. The tools used are constantly evolving, so some parts may be outdated. Please check the documentation of APIs and platforms for the latest changes if you run into issues. Please also consider which parts apply to your personal project and which don't.

Daisy Liu contributed to updating some sections and Yinqdan Lu contributed to the section on running scripts in the background.

## Table of contents

- [Keep in mind](#keep-in-mind)
- [Workflow overview](#workflow-overview)
- [Setting up Google Colab](#setting-up-google-colab)
  - [Colab GitHub integration](#colab-github-integration)
  - [Creating and saving new notebook](#creating-and-saving-new-notebook)
  - [Editing existing notebooks](#editing-existing-notebooks)
- [Working in Colab](#working-in-colab)
  - [Turning on GPU](#turning-on-gpu)
  - [Getting more memory](#getting-more-memory)
  - [Switching between R and Python](#switching-between-r-and-python)
  - [Saving Colab output](#saving-colab-output)
    - [In GDrive](#in-gdrive)
    - [Locally](#locally)
- [Working in GCP](#working-in-gcp)
  - [Signing up for GCP](#signing-up-for-gcp)
  - [Using GCE Deep Learning VM](#using-gce-deep-learning-vm)
    - [Setting up a GCE Deep Learning VM](#setting-up-a-gce-deep-learning-vm)
      - [Via the GCP GUI](#via-the-gcp-gui)
      - [Via the GCP CLI](#via-the-gcp-cli)
    - [Connecting to the VM](#connecting-to-the-vm)
    - [Connecting to GitHub](#connecting-to-github)
    - [Executing a script](#executing-a-script)
      - [In the foreground](#in-the-foreground)
      - [In the background](#in-the-background)
    - [Working in JupyterLab](#working-in-jupyterlab)
    - [Saving output](#saving-output)
      - [Downloading from the SSH-in-browser window, uploading to Drive](#downloading-from-the-ssh-in-browser-window-uploading-to-drive)
      - [Downloading via Jupyter, uploading to Drive](#downloading-via-jupyter-uploading-to-drive)
      - [Uploading to GitHub from JupyterLab](#uploading-to-github-from-jupyterlab)
      - [Uploading to GitHub from your VM shell](#uploading-to-github-from-your-vm-shell)
    - [Modifying GCE VM hardware](#modifying-gce-vm-hardware)
    - [Shutting down VM](#shutting-down-vm)
  - [Using Vertex AI Workbench notebook instances](#using-vertex-ai-workbench-notebook-instances)
    - [Starting notebook instance](#starting-notebook-instance)
      - [Managed notebook](#managed-notebook)
      - [User-managed notebook](#user-managed-notebook)
    - [Connecting notebook instance to GitHub](#connecting-notebook-instance-to-github)
    - [Working with the notebook](#working-with-the-notebook)
    - [Saving GCP output](#saving-gcp-output)
      - [Uploading to GitHub](#uploading-to-github-from-your-notebook-instance)
      - [Saving to VM disk, manually downloading and uploading to Drive](#saving-to-vm-disk-manually-downloading-and-uploading-to-drive)
      - [Using a Google Cloud Storage bucket, then sync the bucket with Drive](#using-a-google-cloud-storage-bucket-then-sync-the-bucket-with-drive)
    - [Modifying notebook instance hardware](#modifying-notebook-instance-hardware)
    - [Shutting down instance](#shutting-down-instance)
- [Working with LLMs](#working-with-llms)
  - [Using Hugging Face](#using-hugging-face)
  - [Using ChatGPT API](#using-chatgpt-api)

## Keep in mind

- Documentation is your friend
  - This tutorial shows some initial steps, but the various documentations from Google on [Colab](https://colab.research.google.com/) and [Google Cloud Platform](https://cloud.google.com/docs) as well as [Hugging Face](https://huggingface.co/docs) and [Open AI](https://platform.openai.com/docs/api-reference/introduction) are essential resources. Learning how to read them takes some time, but it is a valuable investment
  - Forums such as [StackOverflow](https://stackoverflow.com/) can help you find answers
- Be mindful of resource usage
  - Only use GPUs when needed, and only as large as needed
  - Turn VMs/notebook instances off as soon as you don't need them anymore

## Workflow overview

This tutorial lays out the workflow for working on an LLM research project (in this tutorial, we call the project LLM project) and walks you through different steps.

- To ensure reproducibility, traceability and version control
  - If working for an organization, always use your organizations account (e.g., **Stanford account**) when working in Google Colab or Google Cloud Platform (GCP)
  - Always **save your code** &mdash; whether from Colab or GCP &mdash; **to GitHub, at least once per day**, ideally more often
- For resource-intense code, wait for approval in the code review process before you run your code
- Start developing any code using Google Colab
  - Colab is easier to access and free, but has limited resources available and will shutdown automatically at some point
  - When you need more computing power or a faster machine, switch to GCP
- Run resource-intensive code (e.g., in terms of time or memory) in GCP
- Store all output in a dedicated output folder that is accessible to your collaborators
- Document relevant model settings and decisions (e.g., on GitHub via Issues or in the Wiki, or Drive)
- Ask technical questions via [GitHub Issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue) and tag the relevant people as reviewers

## Setting up Google Colab

### Colab GitHub integration

To enable syncing your code easily between Colab and GitHub, follow the following steps:

- Go to http://colab.research.google.com/github
- Check the `Include Private Repos` box
- In the pop-up window, sign in to your GitHub account and give Colab permission to read private files from GitHub

### Creating and saving new notebook

- Create a Colab notebook in Drive and save it in a folder that is accessible to your collaborators, e.g. a `colabs` folder in this example
![image002](https://github.com/user-attachments/assets/c12796b3-0873-41d0-9774-88a8351572e2)
- Give the notebook a descriptive name
- Save your results to GitHub via `File` &rarr; `Save a copy in GitHub`
![image003](https://github.com/user-attachments/assets/d8f4c89a-8608-4706-951a-f3bbdf15cf39)
- Specify the correct repository, branch and file path
- Write a meaningful commit message
![image004](https://github.com/user-attachments/assets/0d76f6af-436e-4d50-a383-51a2540c970f)
- You will then see the Colab in GitHub and can leave the Colab environment

### Editing existing notebooks

- Go to https://colab.research.google.com/github/
- Choose the repository and branch where your notebook is stored and then select `Open notebook in new tab`, the icon at the very right of the selected notebook
- Edit the notebook as you wish
- Save your results to GitHub via `File` &rarr; `Save a copy in GitHub`
- Specify the correct repository, branch and file path
- Write a meaningful commit message
- You will then see the Colab in GitHub and can leave the Colab environment

## Working in Colab

### Turning on GPU

- To speed up the code, you can select to use a GPU as described as shown in the screenshot below
  - Click on the menu at the top right that shows `RAM` and `Disk` (or `Connect` if you haven't connected to a kernel yet)
  - Click on `View Resources`
  - Click `Change runtime type` and select `GPU` as a `Hardware accelerator`
  - Click `Save` and your Colab kernel will restart with the new specification (all variables will be lost)
![image006](https://github.com/user-attachments/assets/f8697ba4-1c26-48c2-ab34-ea2b06af897a)

Don't turn GPU on by default (neither in Colab nor GCP), only use it when you need it. A GPU isn't really necessary for many tasks. It costs more to run a GPU on GCP, it has a higher environmental footprint, and you might see Google Colab complain if you don't use the GPU you added.

### Getting more memory

To get 25GB instead of 13 GB RAM for free, copy the [high memory Colab template](https://colab.research.google.com/drive/1GP_erq_GhbpJmNpg8eS8ghlvlYDpj2Vl).

### Switching between R and Python

You can create a Colab notebook that primarily runs Python or R code by changing the runtime type. E.g., when you want to change the default notebook type from Python to R, click on `Runtime` &rarr; `Change runtime type` and select `R` as `Runtime type`. Note that a change in runtime type will mean that all your variables get lost and you have to rerun the code, thus this is only recommended at the very beginning if you want to set up a notebook to write R code.
![image007](https://github.com/user-attachments/assets/de44daa1-7ced-4835-92a2-fc1809f624f3)

It probably makes sense to create a new notebook/script when you want to switch to another programming language like R since you will likely perform different tasks (e.g., data generation vs. data visualization). However, in case you want to switch between the two in the same notebook, you can use the [`rpy2`](https://rpy2.github.io/) package to do so.

### Saving Colab output

#### In GDrive

- To easily find the folder for your research project, add a shortcut to it to MyDrive.
- Then, from inside your Colab notebook, click the folder icon on the left to see your Drive, locate the folder in which you want to save your file, and click on the three dots on the right of the desired folder and select `Copy path`
- You can then save output using `pd.to_csv()` or other functions as if you were saving to a local machine

#### Locally

Once you saved output in Drive, you can download it from there to your local machine if needed.
![image011](https://github.com/user-attachments/assets/477fccdb-7fa9-4b72-baab-1b6f5d4398db)

## Working in GCP

There are different options to work with GCP VMs and Jupyter notebooks. You can either create a VM with Google Compute Engine directly and SSH into it to start a notebook, or you can start a notebook instance VM via Vertex AI Workbench. Notebook instances come in two flavors, managed and user-managed. There is a trade-off between convenience/ease of use via UI on the one hand, and customizability and cost on the other hand. A GCE VM you configure yourself is the cheapest option and requires some use of the command line. Notebooks are easier to use and don't require any command line interactions beyond GitHub connection. Here is an [article comparing the options](https://www.tensorops.ai/post/what-is-the-best-cloud-jupyter-service-on-google).

Given how much more expensive notebooks can be, especially when you use GPUs, I recommend creating a VM yourself. User-managed notebooks have only relatively small management fees, but they will add up, and they cannot easily run jobs in the background when you leave the notebook environment. Managed notebooks can execute in the background, but the management fees are 10x that of user-managed notebooks. The cost for the underlying GCE VM resources is the same, so you save on management fees by using a GCE VM you set up yourself (see [pricing information](https://cloud.google.com/vertex-ai/pricing#user-managed-notebooks)).

Multiple people can work on the same VM at the same time, for example, in a GCE VM you set up you will notice that each user has their own account on that machine, which others might be able to access.

### Signing up for GCP

Note: This section assumes that a Google Cloud Platform project has already been created by the project lead, and an invitation to the project has been sent to you (the user setting up an account) via your organization email address. It also assumes that the project lead pays for usage via the project account. If you need to create a GCP project from scratch, check out the [GCP project setup documentation](https://developers.google.com/workspace/guides/create-project).

- Navigate to the [GCP homepage](https://cloud.google.com) while being logged into your organization's (e.g., Stanford) email account
- Click on `Start with a full account`
![image012](https://github.com/user-attachments/assets/f8799732-7508-4d18-9bab-e7e0f99d0ddc)
- Agree to the Terms of Service
- Choose `Individual` in the `Account Type` field
![image013](https://github.com/user-attachments/assets/7996a2b2-2340-4c50-9e41-7d869de4c5df)
- Fill in the address and credit card details (if you work for an organization and billing is set up correctly so that bills are sent to your organization's project account, you will not be billed on that account)
- Navigate to the [Google Cloud Console](https://console.cloud.google.com/)
- You should now see the name of your organization's project account in the top menu, or be able to select it in the dropdown menu right next to the Google Cloud logo on the top left
- If you run into errors, make sure you are **logged into your organization's account** because only that account has project access. You might have to sign out of your private or other Google account if that is currently the default

### Using GCE Deep Learning VM

#### Setting up a GCE Deep Learning VM

We create a VM instance using an image to ensure that the necessary libraries are preinstalled. Here, we work with a GCE Deep Learning VM. 

##### Via the GCP GUI

- Navigate to the [Deep Learning VM page in Marketplace](https://console.cloud.google.com/marketplace/product/click-to-deploy-images/deeplearning) and ensure you are in the right project
- Click `LAUNCH` and you will be taken to a window where you can customize your machine
- Add a descriptive name in the `Deployment name`
- Chose your zone. Pick one that is close to where you are phsycially located, e.g. if you are in Stanford, you could choose one that starts with `us-west1` so you stay in the same region. There are various [best practices for location selection](https://cloud.google.com/solutions/best-practices-compute-engine-region-selection), but for this project, there are three main factors to consider. You will probably be fine with the default (that I set [here](https://cloud.google.com/compute/docs/regions-zones/changing-default-zone-region))
  - Availability: sometimes, you might run into errors saying a VM, especially with a GPU, cannot be deployed. Try a different region instead. Further, different regions have different VM options. You can see all [VM options by region](https://cloud.google.com/compute/docs/regions-zones#choosing_a_region_and_zone)
  - Latency: the closer the region is to your actual location, the faster your communication with the remote machine will be. This shouldn't matter so much for this project, but it might be best to stick with US regions
  - Pricing: some regions are cheaper than others 
- All of the following choices depend on your needs, but there are some suggestions based on standard needs
- For `Machine type`, click on the the `General purpose` section 
- For `Series`, pick `N1`
- For `Machine type`, pick a model with as many cores (vCPUs) and memory as you need, e.g. the default `n1-highmem-2`) (see [VM pricing info](https://cloud.google.com/compute/vm-instance-pricing))
- For GPUs
  - Consider if you need a GPU, if not, **delete the `GPU` by clicking the trash can symbol** next to it. That will dramatically reduce cost
  - If you need a GPU to accelerate your hardware, select a smaller one, e.g. the default `NVIDIA T4` or `NVIDIA Tesla K80` if available. For `Number of GPUs`, `1` should suffice. See [GPU pricing](https://cloud.google.com/compute/gpus-pricing)
- For `Framework` choose based on your needs, likely `PyTorch 1.13 (CUDA 11.3, Python 3.10)` or `R 4.2 (Python 3.7)` when working with Python or R, respectively (note: due to a Cloud update, you should only choose Python 3.8 or bigger if possible, otherwise your code might stop working in the future)
- For GPU, check the box at `Install NVIDIA GPU driver automatically on first startup?`
- For `Access to the JupyterLab`, check the box next to `Enable access to JupyterLab via URL instead of SSH.`
- For `Boot Disk`, select the defaults (`Boot disk type` is `Standard Persistent Disk`, `Boot disk size in GB` is `100`) or make changes based on your needs
- Check the box at the bottom to accept the GCP Marketplace Terms of Service
- Click `DEPLOY` (it will take a moment to deploy)

<img width="776" alt="image016" src="https://github.com/user-attachments/assets/04df9e1d-af82-4ee3-a194-a84391f19803">

##### Via the GCP CLI

- See the [Quickstart instructions](https://cloud.google.com/deep-learning-vm/docs/create-vm-instance-gcloud) for more details
- Activate the Cloud shell at the top right by clicking on the shell symbol in the menu next to your profile picture
- Use the `gcloud` command shown below, after necessary modifications as described below, in shell in your [Compute Engine VM instances overview](https://console.cloud.google.com/compute/instances) to run the code

  ```
  export IMAGE_FAMILY="tf-latest-cu92"
  export ZONE="us-west1-b"
  export INSTANCE_NAME="my-new-instance"
  export INSTANCE_TYPE="n1-standard-8"
  gcloud compute instances create $INSTANCE_NAME \
          --zone=$ZONE \
          --image-family=$IMAGE_FAMILY \
          --image-project=deeplearning-platform-release \
          --maintenance-policy=TERMINATE \
          --accelerator="type=nvidia-tesla-v100,count=8" \
          --machine-type=$INSTANCE_TYPE \
          --boot-disk-size=120GB \
          --metadata="install-nvidia-driver=True"
  ```

- The variables and parameters that you will likely modify are:
  - `IMAGE_FAMILY`: The machine image affects the libraries that are preinstalled. See [GCP information on available images](https://cloud.google.com/deep-learning-vm/docs/images). Suggested image following the GUI instructions above: `pytorch-latest-cu118-debian-11-py310`
  - `ZONE`: Geographic location where the VM will be hosted. For this project, the default is `"us-west1-b"`
  - `INSTANCE_NAME`: Name of the VM instance that you will create
  - `INSTANCE_TYPE`: Configuration of the created VM (e.g., memory, CPUs). For more information, see [GCP documentation on machine resources](https://cloud.google.com/compute/docs/machine-resource)
  - `accelerator`: Type and number of GPU accelerators. For this project, the default is `"type=nvidia-tesla-t4,count=1"`
  - `boot-disk-size`: For this project, the default is `100GB`.
- To determine which machine resources are sufficient for your needs, you can use the tool [Can you run it?](https://huggingface.co/spaces/Vokturz/can-it-run-llm)

#### Connecting to the VM

- Navigate to the [Compute Engine VM instances overview](https://console.cloud.google.com/compute/instances) for your project. You should now see your VM there
- Set up the `gcloud` command line interface (CLI) by following the instructions [here](https://cloud.google.com/sdk/docs/install-sdk) (you will need it for easy access to Jupyter notebooks)
- You have two options
  - Using a new browser window (recommended)
    - Click on `Open in browser window` in the `SSH` dropdown menu of your instance
    - This will open a new browser window in which you can now interact with your VM
![image018](https://github.com/user-attachments/assets/4e1bc796-1820-414a-9747-2483a23ed3be)
  - Using `gcloud` command (either in the Google Cloud shell, or a terminal on your local machine, which requires installing the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) on your local machine), which will allow you to SSH into the VM (note that this is more advanced than using the browser UI, but might be preferable e.g. if you needed to upload many files)
    - You can get the `gcloud` command necessary to log into your machine via the command line by clicking on `View gcloud command` in the SSH dropdown menu of your instance
    - Copy the command that should look something like `gcloud compute ssh --zone "us-west1-b" "gce-vm-example-vm" --project "llm-project"`
    - Click `RUN IN CLOUD SHELL` at the bottom of the window with the gcloud command, or, if the window with the gcloud command is already closed, activate the Cloud shell at the top right by clicking on the shell symbol in the menu next to your profile picture
    - Execute the command in the shell and follow the steps shown in the shell to generate a pass key if needed
    - You are now connected to the VM

#### Connecting to GitHub

- Create a GitHub repository for your project (see [GitHub instructions](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)) if necessary. For this tutorial, we assume a repository named `llm_tutorial` has already been created
- In the shell, type `git clone https://github.com/ruthappel/llm_tutorial.git` and authenticate. You now have access to the Git repository
![image024](https://github.com/user-attachments/assets/d7e13433-17cd-4ffc-90b2-4b76e848032d)

#### Executing a script

- Prepare your script and the machine
  - Make sure all packages that are needed to execute the code are installed in the script if they are not pre-installed and you will not install them on the machine
  - You can also install packages directly on the machine (but you will have to do this again every time you start a new VM) using `pip3 install`, e.g. `pip3 intall pandas` if you wanted to install the `pandas` package (this is just an example, `pandas` is already installed)
  - Make sure to save your output at the end to the permanent disk of the VM (i.e., with code as if you were saving to a local machine, such as `pd.to_csv()`)
  - Consider including print statements that help you debug (e.g., printing out at which iteration of a loop you are, see example in the [test_script_long notebook](https://github.com/ruthappel/llm_tutorial/blob/main/test_script_long.py)
  - Ensure the script is a `py` file, e.g., you could create it in Colab and then download as a `py` file
- Upload your script in the SSH-in-browser window (there are other options, like shell commands, listed [here](https://cloud.google.com/compute/docs/instances/transfer-files). Specifically, using `gcloud scp` might be easier for uploading many files, but requires interacting with the Google Cloud CLI or installing the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) on your local machine)
  - Click `UPLOAD FILE` at the top and choose the scripts (files) you want to upload. Let's upload `test_script_short.py` and `test_script_long.py`, which I downloaded as `py` files from the [tutorial folder](https://github.com/ruthappel/llm_tutorial/blob/main/)
  - run `ls` to confirm the files were uploaded successfully (you should also see an upload status window at the bottom right)
![image026](https://github.com/user-attachments/assets/47160a48-ba3f-4120-9916-527e3e478d6b)

##### In the foreground

If it is a short script, you might want to execute it in the foreground and actively see the output. Let's run `test_script_short.py` by typing in the command (note that you might have to enclose the final name in single quotes (`'`) if it contains special characters such as `(`)

```
python3 test_script_short.py
```

We can see that this printed the current working directory to the console, and also a status message of every iteration that it finished
![image027](https://github.com/user-attachments/assets/9c5e1958-fcd4-4f85-bcc3-923e28a13bf9)

#### In the background

If you want to execute a script that takes a long time to run, set your VM up such that it does not stop executing even if the machine shell is closed (e.g., because you accidentally close the browser window or the internet connection is interrupted)
- Run your code using the following code, only replacing `script_name.py` with the name of your script (this is similar to using `slurm/sbatch` on Sherlock)

    ```
    nohup python3 script_name.py 2>&1 &
    ```

  - `nohup` ensures "no hang up", i.e. that your VM will keep running even if you close the shell/browser window
  - `&` at the very end means that your job is executed in the background, i.e., you can still execute other jobs, such as checking in on the status of your jobs (if it is executed in the foreground, a job blocks the console until it finished executing)
  - `2>&1` redirects the standard output that would otherwise be shown in the shell (e.g., print statements) as well as error messages to a nohup.out file that you can inspect later
- Once you execute this command, it will send your task to your VM and print out the process ID (PID), e.g. `14951` in the example below
  - As the process is running in the background, you won't see anything in console, but you can use `ps -u user_name`, replacing `user_name` with the username before the `@` sign before your VM name (`rappel` in the example below) to check in on the currently running processes and see its PID, terminal, time and command
  - Notice that when I executed `ps -u rappel` a second time, the process `14951` was not in the list anymore, and instead we saw a message saying that one process was `Done`, indicating that our script finished running
  - Using `ls` to list the content of the directory, we can confirm that the script generated the expected output `df_result_long.csv` as well as a `nohup.out` file
![image028](https://github.com/user-attachments/assets/f758ed76-edc8-41e9-8269-9120ee8b6bd3)
  - We can open the `nohup.out` file using vim to check if any errors occurred and what the console output that was redirected looked like by typing in `vim nohup.out`
![image029](https://github.com/user-attachments/assets/39c582c2-93d4-43f4-9987-ae379952e71b)
    - All looks good here! No error messages and only the expected output
    - Close `vim` by typing `:wq` and hitting enter
- You can also use `vim` to edit your `py` script directly, but ensure that you check in the final version on GitHub (i.e., download  or directly commit to GitHub the latest version)
- To check your GPU usage, you can run `nvidia-smi`, or run `watch nvidia-smi` to monitor it live
- When you run out of memory, it could be that the trash of your JupyterLab hasn't been emptied yet. You can empty it with code similar to this `rm -rf /home/jupyter/.local/share/Trash`

#### Working in JupyterLab

- Navigate to the GCE Deployment Manager and click on your instance name to see its details
![image030](https://github.com/user-attachments/assets/3744b504-8b52-4aa0-b9e6-7ac4de0fed7f)
![image031](https://github.com/user-attachments/assets/fdf831ff-6dac-4e51-bd39-6aeef2426de1)
- You can copy the code for accessing a Jupyter notebook that is already running on your machine at the righthand side. You need to delete the last part `| grep datalab` for it to work. It should look something like this:
`gcloud compute instances describe --project llm-project --zone us-west1-b gce-vm-example-vm | grep googleusercontent.com`
- Copy this code into your SSH-in-Browser window
![image033](https://github.com/user-attachments/assets/b057bce2-4341-402a-af59-338d6ee71d87)
- Copy the output you get after hitting enter, and use this as a URL to open JupyterLab in your local machine browser
![image034](https://github.com/user-attachments/assets/47088d63-5a64-44d5-96bd-f41bd346651d)
- You can now use JupyterLab just like you would on another notebook instance (see details in the notebook sections below, e.g. how to connect to GitHub via JupyterLab or interact with files there). The link should stay the same for this VM unless you change its settings

#### Saving output

You have many different options to save your output. Unfortunately directly saving output to GDrive is tricky, but any of the methods below should work.

##### Downloading from the SSH-in-browser window, uploading to Drive

- Click on `DOWNLOAD FILE`
- Specify the file path to the file you wish to download to your local machine (e.g., `df_result_long.csv` is in the home directory, so the file path is just the filename, otherwise the directories would be prepended)
- Click `Download` to download
- Then upload this file to the output folder where you store your outputs in GDrive

##### Downloading via Jupyter, uploading to Drive

- If you stored output on the VM using JupyterLab, you will see the output in your JupyterLab file browser on the left. Rightclick the output and select `Download` to download it to your local machine
- Then upload this file to the folder where you store your output in GDrive

##### Uploading to GitHub from JupyterLab

Follow the steps outlined in [Uploading to GitHub from your notebook instance](#uploading-to-github-from-your-notebook-instance).

##### Uploading to GitHub from your VM shell

- If you want to upload via Juptyer, see the section [Uploading to GitHub from your notebook instance](#uploading-to-github-from-your-notebook-instance)
- For uploading from your shell, first navigate to the main respository folder. Use `cd` to change your directory, e.g. `cd llm_tutorial`
- After cloning your repository (which you might have completed in an earlier section, then no need to redo it), you might need to move files around. If that is the case, you can use the `mv origin_path destination_path` command, replacing `origin_path` with the path to the file you want to move and `destination_path` with the path you want to move the file to. You can use `ls` to list the content of a directory to confirm your file was moved
- Enter the following sequence of commands, replacing the `user.name` with your GitHub user name and the `user.email` with the one you can find in your [GitHub email settings](https://github.com/settings/emails) in the `Primary email address` section

    ```
    git config user.name "ruthappel"
    git config user.email "40501125+ruthappel@users.noreply.github.com"
    git add .
    git commit -m 'update gcp tutorial notebook'
    git push
    ```

- When prompted, enter your GitHub user name and authentication token

#### Modifying GCE VM hardware

- You can change the hardware, which will affect performance because you can choose more computing power (e.g., add a GPU, add more CPU cores for parallel tasks) and memory. See [GPU pricing details](https://cloud.google.com/compute/gpus-pricing) and note that the lowest options, such as `NVIDIA Tesla K80` if available or `NVIDIA T4` are usually enough
- Be sure to backup any data you might want to store, they might get lost with hardware modifications
- Navigate to the [Compute Engine VM instances overview](https://console.cloud.google.com/compute/instances) of your project
- Stop the instance you want to modify by checking the box next to the instance name and clicking on `STOP` at the top
- Click on the name of the instance you want to modify. This will open its details page
- Click `EDIT` at the top
- Modify the hardware as desired
- Click SAVE
- Your VM should now be modified

#### Shutting down VM

- **Always shut an instance down when you are done with a session**
- Navigate to the [Compute Engine VM instances overview](https://console.cloud.google.com/compute/instances) of your project
- Select the VM you want to shut down and click `STOP`
![image038](https://github.com/user-attachments/assets/239e83d2-c601-4825-9f08-64c41c27e59b)
- If you will not use it again soon (that is, in a day or so, and there is nothing stored on disk anymore that you need to download), please click on `DELETE` instead. While a stopped machine will not incur costs for CPU and GPU use per time unit, the mere existence of a machine also costs money (e.g., storage costs)
![image039](https://github.com/user-attachments/assets/b4319656-39be-49d6-937f-0243114ac1ff)
- It seems like shutting down the VM in GCE doesn't delete the deployment, so you can additionally go to the [Deployment Manager](https://console.cloud.google.com/dm/deployments) of your project and click on `DELETE` there
![image040](https://github.com/user-attachments/assets/7187c51f-8785-4675-942a-d7a9bff60ce4)
  - Confirm full deletion by selecting `Delete [vm_name] and all resources created by it, such as VMs, load balancers and disks` and clicking `DELETE ALL`
![image041](https://github.com/user-attachments/assets/69b689a8-e98a-42a6-bf59-7919bd7b3c71)

### Using Vertex AI Workbench notebook instances

#### Starting notebook instance

Google Cloud Platform offers a Notebook API that is part of the Vertex AI Workbench. This Notebook API will spin up a VM, but hide a lot of the complexity. Instead of needing to SSH or otherwise connect to the machine, you can go to the GCP UI to launch a JupyterLab notebook that will open up in your browser and allow you to work on the VM via the notebook.

There are two different options: managed and user-managed notebooks. Managed notebooks have some extra features like scheduling executions and shutting down idle instances automatically, and provide a broader choice of notebook types (Python, R, PyTorch) as well as the ability to adapt your resources (CPUs, GPU) without restarting the instance. User-managed notebooks are cheaper, more customizable and geared towards deep learning in terms of the default setup. More details on the choices are outlined [here](https://cloud.google.com/vertex-ai/docs/workbench/notebook-solution).

##### Managed notebook

- Head to the [Workbench UI section for managed notebooks](https://console.cloud.google.com/vertex-ai/workbench/managed) of your project
- Click `CREATE NOTEBOOK` or `NEW NOTEBOOK` at the top
![image042](https://github.com/user-attachments/assets/e86dad98-01d0-4f39-a582-d0d3048c64d4)
- Enter a descriptive `Notebook name`
- Choose a `Region`. There are various [best practices for region selection](https://cloud.google.com/solutions/best-practices-compute-engine-region-selection), but for this project, there are three main factors to consider. You will probably be fine with the default (that I set [here](https://cloud.google.com/compute/docs/regions-zones/changing-default-zone-region))
  - Availability: sometimes, you might run into errors saying a VM, especially with a GPU, cannot be deployed. Try a different region instead. Further, different regions have different VM options. You can see all [VM options by region](https://cloud.google.com/compute/docs/regions-zones#choosing_a_region_and_zone)
  - Latency: the closer the region is to your actual location, the faster your communication with the remote machine will be. This shouldn't matter so much for this project, but it might be best to stick with US regions
  - Pricing: some regions are cheaper than others
- Customize your VM if necessary (e.g., you might want to add a GPU) by clicking on `Advanced Settings`
- Click on `CREATE`
![image043](https://github.com/user-attachments/assets/12fd7a0f-a800-4fd7-b0c1-1de1662757e7)
- This will take you back to the Workbench and the notebook instance will show up (it might take a few seconds to initialize)
- The instance is now running. Click on `OPEN JUPYTERLAB` to launch the notebook

##### User-managed notebook

- Head to the [Workbench UI section for user-managed notebooks](https://console.cloud.google.com/vertex-ai/workbench/user-managed) of your project
- Click `CREATE NOTEBOOK` or `NEW NOTEBOOK` at the top
- Select `Customize...` in the menu that appears
- Give the notebook a descriptive name in the menu that opens up
![image044](https://github.com/user-attachments/assets/ebcfddf3-cb92-4f60-8437-dd8417474993)
- Choose the `Operating sytem` (`Debian 10` seems to have the most choice, so I recommend that)
- Choose the `Environment` (choose a `PyTorch` or `Python 3` environment if you plan to work in Python, or an `R` environment if you plan to work in R)
![image045](https://github.com/user-attachments/assets/c16d8c26-21ae-4fce-a003-df30667eb477)
- Choose the `Machine type`: the default is likely good enough, but you have options to optimize e.g. for memory intensive work. If you add a GPU, a smaller one should suffice (K80 if available, otherwise T4). You will see how dramatically the hardware choice affects pricing. Note that you can also add multiple GPUs, and in some cases, adding multiple smaller GPUs may be less expensive, but as powerful as adding one larger GPU
![image046](https://github.com/user-attachments/assets/06ed84b0-6844-4f14-9ab5-0e4cd0bfb695)
- Choose the `Disks`, i.e. the storage on your VM that you will have access to. The default of 100GB is good
- Keep the the `Networking settings` unless you need to change them
- For `IAM and security`, check `JupyterLab Real Time Collaboration` if you want to collaborate live
- For `System health`, optionally check all boxes under `Reporting` so we get more usage statistics
- Finally, hit `CREATE` at the bottom
- This will take you back to the Workbench and the notebook instance will show up (it might take a few seconds to initialize)
- The instance is now running. Click on `OPEN JUPYTERLAB` to launch the notebook

#### Connecting notebook instance to GitHub

- Click on `Git` &rarr; `Clone a Repository`
![image047](https://github.com/user-attachments/assets/df62628a-ff75-48ed-b786-41aba5ea91b7)
- Enter the URI `https://github.com/ruthappel/llm_tutorial.git`
- Click `Clone`
- Enter your GitHub username and password or access token (GitHub requires two-factor authentication and therefore you will likely need to enter your access token and not your GitHub password)
  - To generate an access token if you don't have one yet, navigate to the [GitHub personal access token site](https://github.com/settings/tokens) and click on `Generate new token` on the top right and select `Generate new token (classic)`
![image048](https://github.com/user-attachments/assets/20b96ac8-7989-4729-a018-a217f7b6241c)
  - In the menu that opens up, ensure that you check the box for `repo` in the scope
  ![image049](https://github.com/user-attachments/assets/03bca863-9ec7-4b84-8439-1dd83253923b)
  - Further information
    - You can find more details about GitHub access tokens [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens))
    - You can find more information on integrating a notebook instance with GitHub [here](https://cloud.google.com/vertex-ai/docs/workbench/user-managed/save-to-github)
- Click `OK`
- Now, the `llm_tutorial` repo should show up in your file system and you can now access all files in the repo

#### Working with the notebook

- Open the notebook you want to run
- For a managed notebook, select a kernel first depending on your needs (most likely `PyTorch`, `Python` or `R`); for a user-managed notebook, the kernel depends on your VM setup earlier. User-managed notebooks only have one kernel that is selected by default
- If an API call doesn't succeed and runs for a very long time, interrupt the kernel (by clicking the grey square at the top) and try running it again
- You can run the notebook just like on Colab or a local machine. You might have to install additional packages if you run into errors when loading packages
- Software that you install will still be available when you start the notebook instance the next time
- If you want to run a longer task and not keep JupyterLab open, you can run a task in the background by SSHing into your notebook instance as described in the section [Executing a script](#executing-a-script) 

#### Saving GCP output

You have multiple options to save your output with GCP.

Unfortunately, a direct [integration with Google Drive is not supported](https://research.google.com/colaboratory/marketplace.html). You also can't mount a Drive on a Colab running on a GCE VM.
There seem to be ways to make it work using a service accounts, but this would require uploading secret credentials or installing extensions. 

We can use three alternative ways described below.

Even if the VM has a permanent disk, **save your output at the end of every session and before you make changes to your hardware** because you might delete your VM or the content on disk might not be preserved when you change settings like the runtime type or GPU.

##### Uploading to GitHub from your notebook instance

- Click on `Git` &rarr; `Open Git Repository in Terminal` in the top menu
- Enter the following sequence of commands, replacing the `user.name` with your GitHub user name and the `user.email` with the one you can find in your [GitHub email settings](https://github.com/settings/emails) in the `Primary email address` section

    ```
    git config user.name "ruthappel"
    git config user.email "40501125+ruthappel@users.noreply.github.com"
    git add .
    git commit -m 'update gcp tutorial notebook'
    git push
    ```

- When prompted, enter your GitHub user name and authentication token

##### Saving to VM disk, manually downloading and uploading to Drive

- Find the current working directory using `os.getcwd()`
- Find (or create) the directory in the project where you want to store the file
- Write the file to this directory just like you did in Colab (e.g., using `pd.to_csv`), only substituting the file path (e.g., `/content/drive/MyDrive/LLM Project/output/tutorial_output/df_chinese_woman_prompt_gpt2.csv` might become `/home/jupyter/llm_tutorial/output/tutorial_output/df_chinese_woman_prompt_gpt2_gcp.csv`)
- Download the output file and upload it to your output folder in GDrive

##### Using a Google Cloud Storage bucket, then sync the bucket with Drive

- Check out the instructions on [StackOverflow](https://stackoverflow.com/questions/48122091/copy-file-from-google-drive-to-google-cloud-storage-within-google), [Medium](https://philipplies.medium.com/transferring-data-from-google-drive-to-google-cloud-storage-using-google-colab-96e088a8c041), [Apache.org](https://airflow.apache.org/docs/apache-airflow/1.10.8/howto/operator/gcp/gcs_to_gdrive.html), or the [External data: Local Files, Drive, Sheets, and Cloud Storage](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=D78AM1fFt2ty) tutorial (see section on GCS)
- More generally, using GCP storage buckets might make sense when you need to access and store large files quickly. This storage is accessible to GCE VM like a local file system, i.e. loads very fast and stores permanently, whereas your storage on a VM can get lost. Google Cloud Storage is a separate product that is created and charged in addition to notebook instances/VMs. You can also access the files from the bucket in your virtual machine, which will require you to mount the data. See this [quick start guide](https://cloud.google.com/storage/docs/gcsfuse-quickstart-mount-bucket) for more instructions

#### Modifying notebook instance hardware

- You can change the hardware, which will affect performance because you can choose more computing power (e.g., add a GPU, add more CPU cores for parallel tasks) and memory. See [GPU pricing details](https://cloud.google.com/compute/gpus-pricing) and note that the lowest options, such as `NVIDIA Tesla K80` if available or `NVIDIA T4` are usually enough
- For managed notebooks, click on the the machine specifications at the top right and select `Modify Hardware Configuration`. This leads ou to a screen where you can modify the instance without relaunching it
- For user-managed notebooks, navigate to the [Workbench](https://console.cloud.google.com/vertex-ai/workbench/user-managed) of your project. Note that the notebook needs to be shut down to be modified
  - Click on the name of the notebook you want to modify. This opens the Notebook details
  - Go to the `HARDWARE` menu
  - Change the settings as desired (see [list of available container images](https://cloud.google.com/deep-learning-containers/docs/choosing-container#listing-versions))
  - Hit `SUBMIT` at the bottom when done

#### Shutting down instance

- **Always shut an instance down when you are done with a session**
- Navigate to the [Workbench instance overview](https://console.cloud.google.com/vertex-ai/workbench/managed) of your project
- Select the instance you want to shut down and click `STOP`
![image057](https://github.com/user-attachments/assets/19e83975-de12-4b01-831d-c3fb8dc1e611)
- If you will not use it again soon (that is, in a day or so, and there is nothing stored on disk anymore that you need to download), please click on `DELETE` instead. The existence of each machine also costs money (e.g., storage costs)

## Working with LLMs

### Using Hugging Face

The notebook [hugging_face_tutorial.ipynb](https://github.com/ruthappel/llm_tutorial/blob/main/hugging_face_tutorial.ipynb) in this repository provides an introduction to working with LLMs via Hugging Face.

### Using ChatGPT API

The notebook [chatgpt_api_tutorial.ipynb](https://github.com/ruthappel/llm_tutorial/blob/main/chatgpt_api_tutorial.ipynb) in this repository provides an introduction to working with LLMs via OpenAI's ChatGPT API.
