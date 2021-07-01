#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=01:00:00

echo "Started"

cd $HOME/MNIST_exercise

module load anaconda3/personal

source activate MNIST_venv

python3 $HOME/MNIST_exercise/main_pytorch.py

echo "Finished"