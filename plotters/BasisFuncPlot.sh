# # Space
# reset
# set grid ytics mytics
# set title "Spatial Basis Functions"
# set key at 0.9,1.6 font ",13"
# set term postscript eps enhanced color blacktext "Helvetica" 18
# set output "BasisFunc_Space.eps"
# plot  "../data/Raw_BasisFunc.txt" using 1:2 with linespoints lt -1 lc 'black' lw 3  ps 1.5 title '1',\
#       "../data/Raw_BasisFunc.txt" using 1:3 with linespoints lt -1 lc 'blue' lw 3  ps 1.5 title '2',\
#       "../data/Raw_BasisFunc.txt" using 1:4 with linespoints lt -1 lc 'green' lw 3  ps 1.5 title '3',\
#       "../data/Raw_BasisFunc.txt" using 1:5 with linespoints lt -1 lc 'red' lw 3  ps 1.5 title '4'


# # Time
# reset
# set grid ytics mytics
# set title "Time Basis Functions"
# set key at 0.015,-1. font ",13"
# set term postscript eps enhanced color blacktext "Helvetica" 18
# set output "BasisFunc_Time.eps"
# plot  "../data/Raw_BasisFunc.txt" using 1:2 with linespoints lt -1 lc 'black' lw 3  ps 1.5 title '1',\
#       "../data/Raw_BasisFunc.txt" using 1:3 with linespoints lt -1 lc 'blue' lw 3  ps 1.5 title '2',\
#       "../data/Raw_BasisFunc.txt" using 1:4 with linespoints lt -1 lc 'green' lw 3  ps 1.5 title '3',\
#       "../data/Raw_BasisFunc.txt" using 1:5 with linespoints lt -1 lc 'red' lw 3  ps 1.5 title '4'
#
# Diffusivity
reset
set grid ytics mytics
set title "Conductivity Basis Functions"
set key at 4.5,1.4 font ",13"
set term postscript eps enhanced color blacktext "Helvetica" 18 background rgb 'white'
set output "BasisFunc_Conductivity.eps"
plot  "../data/Raw_BasisFunc.txt" using 1:2 with linespoints lt -1 lc 'black' lw 3  ps 1.5 title '1',\
      "../data/Raw_BasisFunc.txt" using 1:3 with linespoints lt -1 lc 'blue' lw 3  ps 1.5 title '2',\
      "../data/Raw_BasisFunc.txt" using 1:4 with linespoints lt -1 lc 'green' lw 3  ps 1.5 title '3',\
      "../data/Raw_BasisFunc.txt" using 1:5 with linespoints lt -1 lc 'red' lw 3  ps 1.5 title '4'
