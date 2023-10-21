for j in 01 02 03 04 05 06 07 08 09 10
do
if [ -f ./metric/UNKNOWN/allpop100g100seed$j.out ]
then
	rm ./metric/UNKNOWN/allpop100g100seed$j.out
fi
python3 zdt3.py < ./input/zdt3/100p100g.in
cp ./stats/allpop100g100.out ./metric/UNKNOWN/allpop100g100seed$j.out
done