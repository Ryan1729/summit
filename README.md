# Raylib-rs Template

This is a template for new Rust projects that use [raylib-rs](https://github.com/deltaphc/raylib-rs). This is mainly designed for my (@Ryan1729) own purposes.

That said, I am sharing this publically, so it's worth noting some things about the license. I used the MIT OR Apache-2.0 license for this template since that is the license I plan to apply to my own projects that use this template, by default.

I have no interest in trying to claim ownership over anyone else's projects that started by using this template. Redistribution of the template itself, as an unfilled template, can use the MIT OR Apache-2.0 license.

# Using the template

Basically, if the building/running instructions work then all you need to do is copy this tempalte to a new folder, optionally run `git init`, rename the `rename-me` folder, update the references, the licenses, and this README itself.
See the included `checklist` script for more details if needed.

# Building/Running

1. If you haven't already, [install Rust/Cargo](https://rustup.rs/).
2. Try running `cargo run`. If that starts up program, you're set.
3. If you got an error, refer to the instructions for [raylib-rs](https://github.com/deltaphc/raylib-rs#installation). Some libraries/utilities may need to be installed. Eventually `cargo run` should work. That said as of this writing, I have only tested on Windows and Linux.
    * From a fairly fresh Ubuntu install, I got it working by running the equivalent following, in addition to what I had installed for other reasons. (circa early 2022)
        * `sudo apt-get install curl cmake libglfw3 libglfw3-dev g++`

____

licensed under MIT OR Apache-2.0 at your option
