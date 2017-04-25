## About
GravitySim is a gravity simulator (duh) written in C/C++, using several algorithms to approximate gravity interactions between objects. The only dependency is the [GLFW2 library](https://github.com/glfw/glfw-legacy), you shoud be able to install it using either apt-get on linux or homebrew on OSX. 

##### Installing GLFW2 on OSX
```
brew tap homebrew/versions
brew install --build-bottle --static glfw2
```

## Usage
Compile with ```make mac``` for OSX or ```make linux``` for linux (only tested on ubuntu). Now you should be able to run it using ```./gravitysim```.  
To customize simulation you can call it with different arguments.
```
./gravitysim number_of_galaxies number_of_loops objects_per_galaxy galaxy_size
```  
For instance:  
```./gravitysim 500 10 1000 100``` creates 10 galaxies with 1000 objects per galaxy, diameter 100 pixels and 500 loops.  
```./gravitysim 500 10000 1 1``` creates 10000 galaxies with one object per galaxy, diameter 1 pixel, and 500 loops.   
You could also customize a lot of options using build_config.h file (such as G constant, sd treshold, max speed or max mass), but it requires rebuilding whole project.
Have fun.

## License
[The MIT License (MIT)](http://opensource.org/licenses/mit-license.php)
