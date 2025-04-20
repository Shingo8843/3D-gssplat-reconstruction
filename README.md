3D Gaussian Splatting for 3D Reconstruction of the ShapeNet dataset
wsl 
source ~/nsenv/bin/activate
ns-train splatfacto --data images2/ --max-num-iterations 2000 --viewer.start true

Render: ns-render dataset --load-config outputs/images2/splatfacto/2025-04-09_130659/config.yml --rendered_output_names=rgb

Export: ns-export gaussian-splat --load-config  outputs/images2/splatfacto/2025-04-09_130659/config.yml --output-dir outputs/pointCloud