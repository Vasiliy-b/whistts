
apt update --yes
apt install ffmpeg --yes
apt-get install libcudnn9-cuda-12 --yes

git clone https://github.com/FurkanGozukara/Whisper-WebUI

cd Whisper-WebUI

git reset --hard

git pull

python -m venv venv

source ./venv/bin/activate

python -m pip install --upgrade pip

pip install wheel

echo "Installing requirements"

cd ..

pip install -r requirements.txt

# Show completion message
echo "Virtual environment made and installed properly"

# Keep the terminal open
read -p "Press Enter to continue..."

