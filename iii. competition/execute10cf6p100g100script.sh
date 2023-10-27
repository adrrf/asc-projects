for j in 01 02 03 04 05 06 07 08 09 10
do
if [ -f ./metric/UNKNOWN/allpop40g250seed$j.out ]
then
	rm ./metric/UNKNOWN/allpop40g250seed$j.out
fi
python3 cf6.py < ./input/cf6_16d/40p250g.in
cp ./stats/allpop40g250.out ./metric/UNKNOWN/allpop40g250seed$j.out
done