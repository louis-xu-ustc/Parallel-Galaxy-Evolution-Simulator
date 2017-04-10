<!---
## Key Deadlines
- Mon April 10th (11:59pm) -- Project Proposal Due
- Tue April 25th (11:59pm) -- Project Checkpoint Report Due
- Wed May 10th (9:00am) -- Project pages are made available to judges for finalist selection.
- Thurs May 11th (3pm) -- Finalists announced, presentation time sign ups
- Fri May 12th -- Parallelism Competition Day + Project Party / Final Report Due at 11:59pm
-->

### Project Proposal
<!---The purpose of the proposal is two-fold:

Writing your ideas down forces you to organize your thoughts about your project .
It gives 15-418/618 course staff the ability to verify your plans are of the right scope given our expectations (it also gives us the ability to offer suggestions and help).
Please create a web page for your project. Your project page should contain the following sections and content:
-->
**TITLE**  
Parallel Galaxy Evolution Simulator  
Yunpeng Xu (yunpengx)  
Zhikun Lao (zlao)  

<!---Please provide the title of your project, followed by the names of all team members. Teams may include up to two students. There are no exceptions to this rule.
-->  

**SUMMARY**  
We are going to use CUDA to speedup the galaxy evolution simulator using different algorithms on the NVIDIA GPUs. Our goal is to compare and anylyze the speedup performance of these algorithms.

<!---Summarize your project in no more than 2-3 sentences. Describe what you plan to do and what parallel systems you will be working with. Example one-liners include (you should add a bit more detail):

We are going to implement an optimized Smoothed Particle Hydrodynamics fluid solver on the NVIDIA GPUs in the lab.
We are going port the Go runtime to Blacklight.
We are going to create optimized implementations of sparse-matrix multiplication on both GPU and multi-core CPU platforms, and perform a detailed analysis of both systems' performance characteristics.
We are going to back-engineer the unpublished machine specifications of the GPU in the tablet my partner just purchased.
We are going to implement two possible algorithms for a real-time computer vision application on a mobile device and measure their energy consumption in the lab.
-->

**BACKGROUND**  
Galaxy evolution usually forms beautiful spiral patterns as a result of rotation, colliding and fusion between different galaxies. However, the forming of these patterns needs a large amount of calculation. Our team wants to use what we have learnt in this course to speedup the simulator.  

Basically, galaxy evolution simulator is a force simulation to calculate the gravitational forces and motion of N bodies. There're mainly three methods to solve this problem: brute force program, Barnes-Hut algorithm and Fast Multipole Method. The computational complexity decreases from N^2, N*lgN to N for these three algorithms. Our team will focus on the Barnes-Hut algorithm and Fast Multipole Method, and compare/analyze their speedup performance. 
If time allows, our team also want to try Hamada's new algorithm [1] to solve the N-body simulation problem.  

The basic idea of Barnes-Hut algorithm is to group the nearby particles by using a macro-particle located at the mass-weighted center of the area. If the distance between two groups are sufficiently far away, we can use the mass center in the actual calculation. Quad-tree is the data structure that's suitable to store the information for this problem. Quad-tree is similar to binary tree but with 4 children for each node. The root node represents the whole space and its 4 children can represent the four quadrants (nw, ne, sw and se) of the space. Since tree is a recursive structure, this procedure can be repeated until the leaf of the tree that represents a single body in this problem. The problem can benefit from parallelism in accumulating the gravitational force by traversing the Quad-tree and updating the position based on the exerted gravitational force.  

Fast Multipole Method is different with Barnes-Hut. It computes the potential at every body, not just the force exerted on the body. It also uses more information in each quadrant than the centric mass and total mass. This helps to increase the accuracy but also incurs more calculation of each iteration.

<!---If your project involves accelerating a compute-intensive application, describe the application or piece of the application you are going to implement in more detail. This description need only be a few paragraphs. It might be helpful to include a block diagram or pseudocode of the basic idea. An important detail is what aspects of the problem might benefit from parallelism? And why?
--->

**THE CHALLENGE**   
The first challenge is that how to assign jobs evenly, since the amount of work per body is not uniform in each iteration. The amount of work for a body in a group with multiple of bodies is different with a body that's far away with most of body groups. Also, bodies will move during two iterations. So the cost and communication patterns will also change over time.  

The second challenge is that how to handle colliding between different bodies. Currently, our team hasn't decided how to handle this case. If we need to consider colliding in this problem, it will make it more challenging, because the position update of one body may need to consider other bodies's influence and re-calculate where the body will go after that.

<!---Describe why the problem is challenging. What aspects of the problem might make it difficult to parallelize? In other words, what to you hope to learn by doing the project?

Describe the workload: what are the dependencies, what are its memory access characteristics? (is there locality? is there a high communication to computation ratio?), is there divergent execution?
Describe constraints: What are the properties of the system that make mapping the workload to it challenging?
-->

**RESOURCES**    
Describe the resources (type of computers, starter code, etc.) you will use. What code base will you start from? Are you starting from scratch or using an existing piece of code? Is there a book or paper that you are using as a reference (if so, provide a citation)? Are there any other resources you need, but haven't figured out how to obtain yet? Could you benefit from access to any special machines?

**GOALS AND DELIVERABLES**   
Describe the deliverables or goals of your project.

This is by far the most important section of the proposal:

Separate your goals into what you PLAN TO ACHIEVE (what you believe you must get done to have a successful project and get the grade you expect) and an extra goal or two that you HOPE TO ACHIEVE if the project goes really well and you get ahead of schedule. It may not be possible to state precise performance goals at this time, but we encourage you be as precise as possible. If you do state a goal, give some justification of why you think you can achieve it. (e.g., I hope to speed up my starter code 10x, because if I did it would run in real-time.)
If applicable, describe the demo you plan to show at the parallelism computation (will it be an interactive demo? will you show an output of the program that is really neat? will you show speedup graphs?). Specifically, what will you show us that will demonstrate you did a good job?
If your project is an analysis project, what are you hoping to learn about the workload or system being studied? What question(s) do you plan to answer in your analysis?
Systems project proposals should describe what the system will be capable of and what performance is hoped to be achieved.

**PLATFORM CHOICE**  
Describe why the platform (computer and/or language) you have chosen is a good one for your needs. Why does it make sense to use this parallel system for the workload you have chosen?

**SCHEDULE**  
Produce a schedule for your project. Your schedule should have at least one item to do per week. List what you plan to get done each week from now until the parallelism competition in order to meet your project goals. Keep in mind that due to other classes, you'll have more time to work some weeks than others (work that into the schedule). You will need to re-evaluate your progress at the end of each week and update this schedule accordingly. Note the intermediate checkpoint deadline is April 16th. In your schedule we encourage you to be precise as precise as possible. It's often helpful to work backward in time from your deliverables and goals, writing down all the little things you'll need to do (establish the dependencies!).

**Reference**  
*[1] Hamada, Tsuyoshi, Tetsu Narumi, Rio Yokota, Kenji Yasuoka, Keigo Nitadori, and Makoto Taiji. "42 tflops hierarchical n-body simulations on gpus with applications in both astrophysics and turbulence." In High Performance Computing Networking, Storage and Analysis, Proceedings of the Conference on, pp. 1-12. IEEE, 2009.*  
*[2] http://adl.stanford.edu/cme342/Lecture_Notes_files/lecture13-14_1.pdf*  
*[3] http://www.cs.princeton.edu/courses/archive/fall03/cs126/assignments/barnes-hut.html*


