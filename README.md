# Parallax

Parallax is an interactive viewer for holograms. I refer to [Rob Hocking](https://youtu.be/ZD1in5Zz5ag?si=CKd1F65tU2YnZCjU&t=192) for a very good explanation of the relevant concepts. [Rust](https://rust-lang.org/tools/install/) is the only requirement to run.

It takes a input a folder of images, representing the hogels of either a full or half parallax hologram. Included is this repo is a half parallax hologram in `examples/black_hole_half`, you may visualise it by running `cargo run -r -- ./examples/black_hole_half --mode half`.

Full parallax holograms are too big to be included in this repo, if you want one to test with just contact me. Below is a demo of the viewer on a full parallax hologram:

<video src="demo.mp4" controls preload></video>