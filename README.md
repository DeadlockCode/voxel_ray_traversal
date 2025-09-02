# Voxel Ray Traversal
Welcome to the official repository for this YouTube video:

[![Watch the video](https://img.youtube.com/vi/ztkh1r1ioZo/maxresdefault.jpg)](https://youtu.be/ztkh1r1ioZo)

This repository implements a voxel ray traversal algorithm based on the 1987 paper by Amanatides and Woo, [A Fast Voxel Traversal Algorithm for Ray Tracing](http://www.cse.yorku.ca/~amana/research/grid.pdf).

The core implementation of the traversal algorithm and rendering is in [shaders/traverse.comp](https://github.com/DeadlockCode/voxel_ray_traversal/tree/main/shaders/traverse.comp). You may also be interested in [src/voxelize.rs](https://github.com/DeadlockCode/voxel_ray_traversal/tree/main/src/voxelize.rs) (voxelization of PLY models) and [src/camera.rs](https://github.com/DeadlockCode/voxel_ray_traversal/tree/main/src/camera.rs) (pixel-to-ray matrix setup). Other than that, most of the code is just to get it running.

## Getting Started
Follow the setup guide for [Vulkano](https://github.com/vulkano-rs/vulkano) to make sure you have all the necessary dependencies. After that, it should be as simple as running `cargo run --release` in the root of the repository.

After getting it running you can use the GUI to change some settings and see stats or click on the render to lock the cursor, at which point you can use WASDQE to move, mouse to look around, and scroll to zoom. Then you can hit ESC to unlock the cursor again.

## Branchless Traversal

In the classic traversal loop, each step advances along the axis with the smallest t-value. The logic looks like this:
```GLSL
if (t.x < t.y) {
    if (t.x < t.z) {
        coord.x += ustep.x;
        t.x += delta.x;
    } else {
        coord.z += ustep.z;
        t.z += delta.z;
    }
} else {
    if (t.y < t.z) {
        coord.y += ustep.y;
        t.y += delta.y;
    } else {
        coord.z += ustep.z;
        t.z += delta.z;
    }
}
```
> Note: `ustep` is just to differentiate from the GLSL built-in function `step`, because otherwise the syntax highlighting looks weird.

This branching version is straightforward, but there are ways to make it branchless in hopes of improving performance. The fastest one I've tried rewrites the selection logic using a mask:
```GLSL
uvec3 mask = uvec3(
    t.x < t.y && t.x < t.z,
    t.x >= t.y && t.y < t.z,
    t.x >= t.z && t.y >= t.z
);
coord += ustep * mask;
t += delta * mask;
```
On paper, this avoids conditionals, but in practice it didn't help. On my GPU, it actually ran about 5% slower compared to the branching version. Of course, that result may vary depending on hardware, so it's worth experimenting with if you're curious.

## Contributing

This repository is primarily an educational reference to accompany the video. No new features are planned, and additional commits will only happen if critical bugs or mistakes are found. 

- **Issues**: Bug reports are welcome.  
- **Pull requests**: Only considered for critical bug fixes.  
- **Feature requests**: Not in scope for this project.