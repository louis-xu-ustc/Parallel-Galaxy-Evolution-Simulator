<!---
## Key Deadlines
- Mon April 10th (11:59 pm) -- Project Proposal Due
- Tue April 25th (11:59 pm) -- Project Checkpoint Report Due
- Wed May 10th (9:00am) -- Project pages are made available to judges for finalist selection.
- Thurs May 11th (3 pm) -- Finalists announced, presentation time sign ups
- Fri May 12th -- Parallelism Competition Day + Project Party / Final Report Due at 11:59pm
-->

## Project Proposal
<!---The purpose of the proposal is two-fold:

Writing your ideas down forces you to organize your thoughts about your project.
It gives 15-418/618 course staff the ability to verify your plans are of the right scope given our expectations (it also gives us the ability to offer suggestions and help).
Please create a web page for your project. Your project page should contain the following sections and content:
-->
### TITLE  

Parallel Galaxy Evolution Simulator  
Yunpeng Xu (yunpengx)  
Zhikun Lao (zlao)  

<!---Please provide the title of your project, followed by the names of all team members. Teams may include up to two students. There are no exceptions to this rule.
-->  

### SUMMARY  

We are going to use CUDA to speedup the galaxy evolution simulator using different algorithms on the NVIDIA GPUs. Our goal is to compare and analyze the speedup performance of these algorithms.

<!---Summarize your project in no more than 2-3 sentences. Describe what you plan to do and what parallel systems you will be working with. Example one-liners include (you should add a bit more detail):

We are going to implement an optimized Smoothed Particle Hydrodynamics fluid solver on the NVIDIA GPUs in the lab.
We are going port the Go runtime to Blacklight.
We are going to create optimized implementations of sparse-matrix multiplication on both GPU and multi-core CPU platforms, and perform a detailed analysis of both systems' performance characteristics.
We are going to back-engineer the unpublished machine specifications of the GPU in the tablet my partner just purchased.
We are going to implement two possible algorithms for a real-time computer vision application on a mobile device and measure their energy consumption in the lab.
-->

### BACKGROUND  

Galaxy evolution simulation usually forms beautiful spiral patterns as a result of rotation, collision, and fusion between different galaxies. However, the forming of these patterns needs a large amount of calculation. Our team wants to use what we have learned in this course to speedup such simulators.

Basically, galaxy evolution simulator is based on force simulation to calculate the gravitational forces and motion of N bodies. There're mainly three methods to solve this problem: brute force, Barnes-Hut algorithm and Fast Multipole Method. The computational complexity decreases from N^2, N*lgN to N for these three algorithms. Our team will focus on the Barnes-Hut algorithm and Fast Multipole Method, and compare/analyze their speedup performance.
Time allowing, our team also want to try Hamada's new algorithm [1] to solve the N-body simulation problem.

The basic idea of the Barnes-Hut algorithm is to group the nearby particles by using a macro-particle located at the mass-weighted center of the area. If the distance between two groups is sufficiently far away, we can use the mass center in the actual calculation. Quad-tree is the data structure that's suitable to store the information for this problem. A quad-tree is similar to a binary tree but with 4 children for each node. The root node represents the whole space and its 4 children can represent the four quadrants (NW, NE, SW and SE) of the space. Since the tree is a recursive structure, this procedure can be repeated until the leaf of the tree that represents a single body in this problem. The problem can benefit from parallelism in accumulating the gravitational force by traversing the Quad-tree and updating the position based on the exerted gravitational force.  

Fast Multipole Method is different from Barnes-Hut. It computes the potential at every body, not just the force exerted on the body. It also uses more information in each quadrant than the centric mass and total mass. This helps to increase the accuracy but also incurs more calculation of each iteration.

<!---If your project involves accelerating a compute-intensive application, describe the application or piece of the application you are going to implement in more detail. This description need only be a few paragraphs. It might be helpful to include a block diagram or pseudocode of the basic idea. An important detail is what aspects of the problem might benefit from parallelism? And why?
--->

### THE CHALLENGE

The first challenge is that how to assign jobs evenly, since the amount of work per body is not uniform in each iteration. The amount of work for a body in a group with multiple of bodies is different from a body that's far away with most of the body groups. Also, bodies will move during two iterations. So the cost and communication patterns will also change over time.

The second challenge is that how to handle collision between different bodies. Currently, our team hasn't decided how to handle this case. If we need to consider collision in this problem, it will make it more challenging, because the position update of one body may need to consider other bodies' influence and re-calculate where the body will go after that.

<!---Describe why the problem is challenging. What aspects of the problem might make it difficult to parallelize? In other words, what to you hope to learn by doing the project?

Describe the workload: what are the dependencies, what are its memory access characteristics? (is there locality? is there a high communication to computation ratio?), is there divergent execution?
Describe constraints: What are the properties of the system that make mapping the workload to it challenging?
-->

### RESOURCES

<!--
Describe the resources (type of computers, starter code, etc.) you will use. What code base will you start from? Are you starting from scratch or using an existing piece of code? Is there a book or paper that you are using as a reference (if so, provide a citation)? Are there any other resources you need, but haven't figured out how to obtain yet? Could you benefit from access to any special machines?

For the brute force approach, we will start from scratch since it is rather straightforward. We will implement two versions: CPU version and GPU version. The CPU version will serve as a baseline implementation for the GPU version. Also, these two implementation will, altogether, be two of our baseline implementations for the Barnes-Hut algorithm approach and the Fast Multipole Method approach.

For the Barnes-Hut algorithm approach, we will use [this](https://github.com/kgabis/gravitysim) as our starter code. This piece of source code lienced under The MIT License. However, this piece is just a CPU sequential version of the Barnes-Hut algorithm. After carefully analyzing and examining its code base, we will create our CPU **baseline implementation** based on it. Adding to that, we will create our GPU version. We will read several papers on this algorithm to aid our understanding.

For the Fast Multipole Method approach, we will start from scratch because it is not as straightforward as the former two methods and we think we can understand it better by coding it. There will also be two versions. 
-->


1. Except for the [sequential version](https://github.com/kgabis/gravitysim) (MIT license) of the Barnes-Hut algorithm, we will start from scratch. We will read several papers on the Barnes-Hut algorithm and the Fast Multipole Method approach. Those papers are listed in the reference section below.
2. We will be using CUDA API
3. We will be using GLFW graphical library, a multi-platform library for OpenGL
4. Some ghc machines have the GeForce GTX 1080 that we need

### GOALS AND DELIVERABLES   

<!--
Describe the deliverables or goals of your project.

This is by far the most important section of the proposal:

Separate your goals into what you PLAN TO ACHIEVE (what you believe you must get done to have a successful project and get the grade you expect) and an extra goal or two that you HOPE TO ACHIEVE if the project goes really well and you get ahead of schedule. It may not be possible to state precise performance goals at this time, but we encourage you be as precise as possible. If you do state a goal, give some justification of why you think you can achieve it. (e.g., I hope to speed up my starter code 10x, because if I did it would run in real-time.)
If applicable, describe the demo you plan to show at the parallelism computation (will it be an interactive demo? will you show an output of the program that is really neat? will you show speedup graphs?). Specifically, what will you show us that will demonstrate you did a good job?
If your project is an analysis project, what are you hoping to learn about the workload or system being studied? What question(s) do you plan to answer in your analysis?
Systems project proposals should describe what the system will be capable of and what performance is hoped to be achieved.

-->

**Plan To Achieve**

1. CPU/GPU Barnes-Hut algorithm
2. CPU/GPU Fast Multipole Method

**Hope To Achieve**

1. Hamada's new algorithm (improvement on the GPU versions of both algorithms)
2. Formation of new planets under planet collision

<!-- 3. interactive user interface -->

<!--
Performance goals:

XXX

-->

**Demo**

We will possibly demonstrate a scene of galaxy collision where thousands of planets of multiple galaxies collide together. Real-time statistics will be shown in the screen, reflecting the number of frames per second and other staticstics. We will also show our speedup graphs which compare the performance of different versions of algorithms mentioend above.

<!--
In order to show that we did a good job, we will ...
-->

<!--
do a calculation of how much speedup ideally we can achieve using that piece of hardware. Then we will compare this ideal speedup with the speedup we achieve.

The factors taken into consideration in the calculation are, for example, hardwre specifications, sequential part, and etc. 
-->

<!--
To enable an interactive interface, we will make use the opengl function which allows users to observe the scene from different angles.
-->

<!--
Adding to that, we may allow users to interact with the collision scene by swiping the pointer (of the mouse) in the scene, where the pointer serves as a gravitational concentration.
-->

### PLATFORM CHOICE  

<!--
Describe why the platform (computer and/or language) you have chosen is a good one for your needs. Why does it make sense to use this parallel system for the workload you have chosen?
-->

We will be implementing the simulator in C++ using CUDA platform to work with GeForce GTX 1080. As far as we know, this is the best GPUs we can get from CMU.

We think this parallel system is suitable for our workload because our workload can be largely run in parallel. For example, in the Barnes-Hut algorithm, during each iteration, the forces applying on all particles can be potentially computed at the same time, as long as we stay within resource limit. Another example is for the computation of Fast Multipole Method. The dynamic parallelism available in CUDA API can help avoid subsequent index computations and achieve good performance. Although one "fits all" kernel is not possible for FMM and we will have to take a hybrid approach, this parallel platform is still our best choice so far.

### SCHEDULE  

| Date        | Goals                                                                                                                                  | Status |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------|--------|
| 4.10 - 4.16 | Refine the starter sequential implementation of the Barnes-Hut algorithm; Complete parallel implementation of the Barnes-Hut algorithm |        |
| 4.17 - 4.23 | Complete the sequential implementation of the Fast Multipole Method                                                                    |        |
| 4.23 - 4.24 | Conduct implementation refinement and complete project checkpoint report                                                               |        |
| 4.25 - 5.2  | Complete the parallel implementation of the Fast Multipole Method                                                                      |        |
| 5.3 - 5.8   | Complete any good-to-have features                                                                                               |        |
| 5.9 - 5.10  | Draft final report and complete project pages                                                                                          

### Reference  

*[1] Hamada, Tsuyoshi, Tetsu Narumi, Rio Yokota, Kenji Yasuoka, Keigo Nitadori, and Makoto Taiji. "42 tflops hierarchical n-body simulations on GPUs with applications in both astrophysics and turbulence." In High Performance Computing Networking, Storage and Analysis, Proceedings of the Conference on, pp. 1-12. IEEE, 2009.*
*[2] http://adl.stanford.edu/cme342/Lecture_Notes_files/lecture13-14_1.pdf*  
*[3] http://www.cs.princeton.edu/courses/archive/fall03/cs126/assignments/barnes-hut.html*  
*[4] https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf*   
*[5] http://on-demand.gputechconf.com/gtc/2015/presentation/S5548-Bartosz-Kohnke.pdf*   
*[6] http://beltoforion.de/article.php?a=barnes-hut-galaxy-simulator*

