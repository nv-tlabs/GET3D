# conda install libgcc
g++ -Wall -fPIC -O2 -c run.cpp -std=c++11 -fpermissive
g++ -shared -o run.so run.o