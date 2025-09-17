#!/bin/bash

module load gnuplot

# Check if OUTCAR exists
if [ ! -f OUTCAR ]; then
    echo "OUTCAR not found in current directory."
    exit 1
fi

# Extract temperature data
grep -i temperature OUTCAR > temp.dat

# Remove first two lines (headers)
sed -i '1,2d' temp.dat

# Generate Gnuplot script
cat << EOF > temp_plot.gnuplot
set title "Temperature vs. Step"
set xlabel "Step"
set ylabel "Temperature (K)"
set grid
plot 'temp.dat' using 6 with lines title 'Temperature'
pause -1 "Press Enter to exit"
EOF

# Run gnuplot
gnuplot temp_plot.gnuplot
