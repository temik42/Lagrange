#define SCALE %(scale)f
#define BLOCK_SIZE %(block_size)d
#define NL (%(block_size)d+2)
#define NX %(nx)d
#define NY %(ny)d
#define NZ %(nz)d
    
#include <math.cl>

__constant float gamma = 5./3;
__constant unsigned int ng[3] = {NX,NY,NZ};
__constant unsigned int bg[3] = {NY*NZ,NZ,1};

struct q_str {
    float3 dX[3],dXb[3],d2X[6],dV[3],B,dB[3];
    float det_dX,d_det_dX[3],G[6],T,dT[3];
};


void __attribute__((overloadable)) 
toPrivate(__global float* X, float* Xp)
{      
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    char i,j,k;

    for (i = -1; i < 2; i++) 
        for (j = -1; j < 2; j++) 
            for (k = -1; k < 2; k++)  
                if ((ii[0]+i>=0) && (ii[0]+i<=ng[0]-1) && (ii[1]+j>=0) && (ii[1]+j<=ng[1]-1) && (ii[2]+k>=0) && (ii[2]+k<=ng[2]-1)) 
                    Xp[jdx + i*bp[0] + j*bp[1] + k*bp[2]] = X[idx + i*bg[0] + j*bg[1] + k*bg[2]]; 
}


void __attribute__((overloadable)) 
toPrivate(__global float3* X, float3* Xp)
{      
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    char i,j,k;

    for (i = -1; i < 2; i++) 
        for (j = -1; j < 2; j++) 
            for (k = -1; k < 2; k++)  
                if ((ii[0]+i>=0) && (ii[0]+i<=ng[0]-1) && (ii[1]+j>=0) && (ii[1]+j<=ng[1]-1) && (ii[2]+k>=0) && (ii[2]+k<=ng[2]-1)) 
                    Xp[jdx + i*bp[0] + j*bp[1] + k*bp[2]] = X[idx + i*bg[0] + j*bg[1] + k*bg[2]]; 
}



struct q_str __attribute__((overloadable)) 
    Forward(float3* Xp, float3* Vp, float* Tp, float* Bp)
{
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};

    float B, dB[3], T, dT[3];
    
    struct q_str Q; 
    uchar i,j;

    jacobian(Xp, Q.dX);
    
    hessian(Xp, Q.d2X);
    jacobian(Vp, Q.dV);
    jacobian(Tp, dT);  
    //jacobian(Tp, Q.dT);    
    //Q.T = Tp[jdx];
    T = Tp[jdx];
    jacobian(Bp, dB);
    B = Bp[jdx];
    
    Q.dX[0].x += 1;
    Q.dX[1].y += 1;
    Q.dX[2].z += 1;
    
    //-----------------------------the rest of code uses only private memory-------------------------------------
    

    Q.det_dX = Q.dX[0].x*Q.dX[1].y*Q.dX[2].z + Q.dX[0].y*Q.dX[1].z*Q.dX[2].x + Q.dX[0].z*Q.dX[1].x*Q.dX[2].y - 
           Q.dX[0].x*Q.dX[1].z*Q.dX[2].y - Q.dX[0].y*Q.dX[1].x*Q.dX[2].z - Q.dX[0].z*Q.dX[1].y*Q.dX[2].x;
    
    Q.dXb[0] = Cross(Q.dX[1], Q.dX[2])/Q.det_dX;
    Q.dXb[1] = Cross(Q.dX[2], Q.dX[0])/Q.det_dX;
    Q.dXb[2] = Cross(Q.dX[0], Q.dX[1])/Q.det_dX;
    
    for (i = 0; i < 3; i++) 
        for (j = 0; j <= i; j++) Q.G[mm(i,j)] = Dot(Q.dXb[i],Q.dXb[j]);   
    
    
    Q.B = Q.dX[2]*B/Q.det_dX;
    
    for (i = 0; i < 3; i++) {
        Q.d_det_dX[i] = 0.;   
        for (j = 0; j < 3; j++) 
            Q.d_det_dX[i] += Q.d2X[mm(i,j)].x*Q.dXb[j].x + Q.d2X[mm(i,j)].y*Q.dXb[j].y + Q.d2X[mm(i,j)].z*Q.dXb[j].z;
        Q.d_det_dX[i] *= Q.det_dX;
    }
    
    Q.T = T*pow(Q.det_dX, gamma-1);
    for (i = 0; i < 3; i++) Q.dT[i] = dT[i]*pow(Q.det_dX, gamma-1) + Q.d_det_dX[i]*Q.T*(gamma-1)*pow(Q.det_dX, gamma-2);  
    
       
    for (i = 0; i < 3; i++) {
        Q.dB[i] = - Q.B*Q.d_det_dX[i] + Q.dX[2]*dB[i] + Q.d2X[mm(i,2)]*B;        
        //Q.dB[i] /= (float3)(Q.det_dX);
    }
    return Q;
}


__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE)))
void Reverse(__global float3* Xg, __global float3* Bg, __global float3* Bg0)
{       
    
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    float3 Xp[27],Bp[27],B,B0;
    
    toPrivate(Xg,Xp);
    toPrivate(Bg,Bp);
    
    B = Bp[jdx];

    float3 dX[3], dB[3], dXb[3];
    
    jacobian(Xp, dX);
    jacobian(Bp, dB);
    
    float det_dX;
    
    dX[0].x += 1;
    dX[1].y += 1;
    dX[2].z += 1;
    
    det_dX = dX[0].x*dX[1].y*dX[2].z + dX[0].y*dX[1].z*dX[2].x + dX[0].z*dX[1].x*dX[2].y - 
           dX[0].x*dX[1].z*dX[2].y - dX[0].y*dX[1].x*dX[2].z - dX[0].z*dX[1].y*dX[2].x;    
    
    dXb[0] = Cross(dX[1], dX[2])/(float3)(det_dX);
    dXb[1] = Cross(dX[2], dX[0])/(float3)(det_dX);
    dXb[2] = Cross(dX[0], dX[1])/(float3)(det_dX);
    
    B0.x = Dot(dXb[0],B);
    B0.y = Dot(dXb[1],B);
    B0.z = Dot(dXb[2],B);
    
    B0 *= (float3)det_dX;
    
    Bg[idx] = B0;
    
    barrier(CLK_GLOBAL_MEM_FENCE);
}


float3 Lorenz(struct q_str Q) {
    
    float3 out;
    float3 current = 0;
    
    uchar i;
    
    for (i = 0; i < 3; i++) {
        current += Cross(Q.dXb[i], Q.dB[i]);  //не забудь поделить на 4*pi!!!
    }
    out = Cross(current, Q.B);//*Q.det_dX;
    return out;
}


float3 Pressure(struct q_str Q) {
    
    float3 gradT = 0;
    float3 grad_det_dX = 0;
    
    uchar i,j;
    
    for (i = 0; i < 3; i++) 
        for (j = 0; j < 3; j++) {
            gradT += Q.G[mm(i,j)]*Q.dT[j]*Q.dX[i];
            grad_det_dX += Q.G[mm(i,j)]*Q.d_det_dX[j]*Q.dX[i];
        }
    
    return (grad_det_dX*Q.T/Q.det_dX-gradT);
}





__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE))) 
void Step(__global float3* Xg, __global float3* Vg, __global float3* DVg, __global float* Tg,
          __global float* Bg)
{       
    
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    float3 Xp[27],Vp[27];
    float Tp[27],Bp[27];
    
    toPrivate(Xg,Xp);
    toPrivate(Bg,Bp);
    toPrivate(Vg,Vp);
    toPrivate(Tg,Tp);

    
    struct q_str Q;
    Q = Forward(Xp, Vp, Tp, Bp);
    
    DVg[idx] = Lorenz(Q) + Pressure(Q)*0.02f;
    barrier(CLK_GLOBAL_MEM_FENCE);
}


__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE))) 
void Increment(__global float3* Xg, __global float3* X0g, __global float3* DXg, 
               __global float3* Vg, __global float3* V0g, __global float3* DVg, float a)
{       
    
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    Xg[idx] = X0g[idx] + DXg[idx]*a;
    Vg[idx] = V0g[idx] + DVg[idx]*a;
    barrier(CLK_GLOBAL_MEM_FENCE);
}


__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE))) 
void Put(__global float3* Xg, __global float3* Vg, unsigned int ind, uchar dim)
{       
    
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    unsigned int jdx;
        
    if (dim == 0) jdx = ii[1]*NZ + ii[2];
    if (dim == 1) jdx = ii[0]*NZ + ii[2]; 
    if (dim == 2) jdx = ii[0]*NY + ii[1];
    
    if (ii[dim] == ind) Xg[idx] = Vg[jdx];
    
    barrier(CLK_GLOBAL_MEM_FENCE);
}


__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE))) 
void Take(__global float3* Xg, __global float3* Vg, unsigned int ind, uchar dim)
{       
    
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    unsigned int jdx;
        
    if (dim == 0) jdx = ii[1]*NZ + ii[2];
    if (dim == 1) jdx = ii[0]*NZ + ii[2]; 
    if (dim == 2) jdx = ii[0]*NY + ii[1];
    
    if (ii[dim] == ind) Vg[jdx] = Xg[idx];
    
    barrier(CLK_GLOBAL_MEM_FENCE);
}



