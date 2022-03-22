numg=1000*[1 8];
deng=poly([-7, -9]);
G=tf(numg,deng);
Kp=dcgain(G)
estep=1/(1+Kp)
T=feedback(G,1);
poles=vpa(pole(T),4)