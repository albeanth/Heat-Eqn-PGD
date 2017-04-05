reset
set logscale y
set format y "%.1e"
set grid ytics mytics
set xlabel "Enrichment Number"
set ylabel "(Abs. Diff.)^2"
set key at 22,3e-6 font ",13"
set term postscript eps enhanced color blacktext "Helvetica" 18
set output "HeatComp.eps"
plot  "../data/ComparisonOfConvergence.txt" using 1:2 with linespoints lt 1 lc 'blue' lw 3  ps 1. title 'TA-Raw', \
      "../data/ComparisonOfConvergence.txt" using 1:3 with linespoints lt 1 lc 'green' lw 3  ps 1. title 'FC-Raw', \
      "../data/ComparisonOfConvergence.txt" using 1:4 with linespoints lt 1 lc 'red' lw 3  ps 1. title 'TA-Comp.', \
      "../data/ComparisonOfConvergence.txt" using 1:5 with linespoints lt 1 lc 'black' lw 3  ps 1. title 'FC-Comp.'
