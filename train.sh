# ROLL training process

if [ $1 == "knowit" ] || [ $1 == "tvqa" ];
then
    # First, each branch is pretrained independently
    python Source/branch_read.py --dataset $1
else
    echo "Undefined parameter. Use 'knowit' or 'tvqa' as \$1 parameter."
fi

