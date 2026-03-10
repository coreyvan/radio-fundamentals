# Raspberry Pi Setup

These notes describe the target runtime for this notebook set: a Raspberry Pi 4 running headless JupyterLab, with the browser UI accessed remotely from another machine on the local network.

## System packages

Install the packages needed for Python environments and media conversion:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg
```

`ffmpeg` is required for loading non-WAV audio inputs such as `.m4a` and `.mp3`.

## Project environment

Clone the repository, then create the local virtual environment inside the project root:

```bash
cd ~/radio-fundamentals
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional later dependency for live SDR work:

```bash
python -m pip install pyrtlsdr
```

The notebooks already gate live RTL-SDR behavior behind `probe_rtlsdr()`, so `pyrtlsdr` is not required for the synthetic-fallback workflow.

## JupyterLab configuration

Generate config once:

```bash
source .venv/bin/activate
jupyter lab --generate-config
```

Set the following in `~/.jupyter/jupyter_lab_config.py`:

```python
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.open_browser = False
c.ServerApp.port = 8888
```

Then set a password:

```bash
source .venv/bin/activate
jupyter lab password
```

## Headless access

After JupyterLab is running, connect from another machine on the local network at:

```text
http://<pi-hostname>.local:8888
```

Widget-backed notebooks depend on `ipywidgets` and `ipympl`, both already included in `requirements.txt`.

## Systemd service

Sample unit file at `/etc/systemd/system/jupyter.service`:

```ini
[Unit]
Description=Jupyter Lab
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/radio-fundamentals
ExecStart=/home/pi/radio-fundamentals/.venv/bin/jupyter-lab
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable jupyter
sudo systemctl start jupyter
```

## Local-only assets

Do not store recordings or captures in git. Put them under `assets/local/` using the filenames documented in [`assets/capture_manifest.md`](./assets/capture_manifest.md).

If a local file is missing, the notebooks fall back to synthetic demo signals instead of failing.
