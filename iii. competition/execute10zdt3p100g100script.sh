for j in 01 02 03 04 05 06 07 08 09 10
do
if [ -f ./metric/UNKNOWN/allpop200g50seed$j.out ]
then
	rm ./metric/UNKNOWN/allpop200g50seed$j.out
fi
python3 zdt3.py < ./input/zdt3/200p50g.in
cp ./stats/allpop200g50.out ./metric/UNKNOWN/allpop200g50seed$j.out
done