lr_g_values=(0.002 0.001 0.0005 0.0001)

# Loop over each lr_g value
for lr_g in "${lr_g_values[@]}"
do
  echo "Running training with lr_g=$lr_g"
  python3 train.py --epochs 40 --lr_g "$lr_g" --lr_d 0.0002
done