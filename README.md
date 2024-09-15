# Advanced project in CSE (Solving Hamilton-Jacobi-Bellman equations)

This repository was developed as part of the advanced project in the course of study CSE. The aim is to solve one and two dimensional Hamilton-Jacobi-Bellman equations (HJB) of order one and two. 

This solver for the HJB has the goal that it should be ease of use. More details about it can be read in the documentation which is located at <mark>/docs/build/html</mark>. Just open the <mark>index.html</mark>  file with the web browser of your choice.

There are three examples in this repository located at <mark>/examples/</mark>. They can be run with the <mark>run_examplex.py</mark> script. The examples are an easy one dimensional HJB of order one. An easy two dimension HJB of order two and a more complicated two dimensional HJB of order two. More about the examples can be read at the documentation.

**NOTE**

If you are running the examples with VS-Code make sure that you are running it with the following seetings in your *seetings.json* and/or *laun.json* file.

- For windows:
`"terminal.integrated.env.windows": { "PYTHONPATH": "${workspaceFolder}" }`

- For linux:
`"terminal.integrated.env.linux": { "PYTHONPATH": "${workspaceFolder}" }`