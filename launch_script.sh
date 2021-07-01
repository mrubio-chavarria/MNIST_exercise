#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=01:00:00

echo "Started"

cd $HOME/MNIST_exercise

source venv/bin/activate

python3 $HOME/MNIST_exercise/main_pytorch.py

echo "Finished"