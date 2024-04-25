dir=/home/ILSVRC2012/train

for x in $dir/*.tar; do
    filename=$(basename $x .tar)
    mkdir $dir/$filename
    tar -xvf $x -C $dir/$filename
    rm $x
done
