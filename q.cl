#define SCALE %(scale)f
#define BLOCK_SIZE %(block_size)d
#define NL (%(block_size)d+2)
#define NX %(nx)d
#define NY %(ny)d
#define NZ %(nz)d
    
#define mm(i,j) (i>j)?i*(i+1)/2+j:j*(j+1)/2+i //"smart" index for triagngular matrices
    
__constant unsigned int ng[3] = {NX,NY,NZ};
__constant unsigned int bg[3] = {NY*NZ,NZ,1};
__constant unsigned int bl[3] = {NL*NL,NL,1};
__constant unsigned int bp[3] = {9,3,1};

struct rb_str {
    float3 force,current,field;
};


struct q_str {
    float3 dX[3],dXb[3],d2X[6],B,dB[3],Bx,dBx[3];
    float det_dX,d_det_dX[3],dT[3];
};


float3 Cross(float3 U, float3 V)
{
    float3 Y;
    Y.x = U.y*V.z - U.z*V.y;
    Y.y = U.z*V.x - U.x*V.z;
    Y.z = U.x*V.y - U.y*V.x;
    
    return Y;
}



float __attribute__((overloadable)) 
Deriv(__local float* Xl, unsigned int ldx, uchar dim, uchar order)
{  
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
       
    float out = 0;
    
    if ((ii[dim] != 0) && (ii[dim] != (ng[dim]-1))) { 
        if (order == 1) out = Xl[ldx+bl[dim]]*(float)(0.5) - Xl[ldx-bl[dim]]*(float)(0.5);
        if (order == 2) out = Xl[ldx+bl[dim]] + Xl[ldx-bl[dim]] - (float)(2)*Xl[ldx];              
    } 
    return out;
}


float3 __attribute__((overloadable)) 
Deriv(__local float3* Xl, unsigned int ldx, uchar dim, uchar order)
{  
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
       
    float3 out = (float3)(0);
      
    if ((ii[dim] != 0) && (ii[dim] != (ng[dim]-1))) { 
        if (order == 1) out = Xl[ldx+bl[dim]]*(float3)(0.5) - Xl[ldx-bl[dim]]*(float3)(0.5);
        if (order == 2) out = Xl[ldx+bl[dim]] + Xl[ldx-bl[dim]] - (float3)(2)*Xl[ldx];              
    } 
    return out;
}


void __attribute__((overloadable)) 
jacobian(float3* Xp, float3* dX)
{  
    unsigned int jj[3] = {1,1,1};
    unsigned int jdx = jj[0]*bp[0]+jj[1]*bp[1]+jj[2]*bp[2];
    
    uchar i;
    
    for (i = 0; i < 3; i++) 
        dX[i] = (Xp[jdx+bp[i]] - Xp[jdx-bp[i]])*(float3)(0.5);
          
}


void __attribute__((overloadable)) 
hessian(float3* Xp, float3* d2X)
{  
    unsigned int jj[3] = {1,1,1};
    unsigned int jdx = jj[0]*bp[0]+jj[1]*bp[1]+jj[2]*bp[2];
    
    uchar i,j;
    
    for (i = 0; i < 3; i++) {
        d2X[mm(i,i)] = Xp[jdx+bp[i]] + Xp[jdx-bp[i]] - Xp[jdx]*(float3)(2);
        for (j = 0; j < i; j++) {
            d2X[mm(i,j)] = (Xp[jdx+bp[i]+bp[j]] + Xp[jdx-bp[i]-bp[j]] - Xp[jdx+bp[i]-bp[j]] - Xp[jdx-bp[i]+bp[j]])*(float3)(0.25);
        }
    }
}



void __attribute__((overloadable)) 
toLocal(__global float* X, __local float* Xl)
{      
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    unsigned int ll[3] = {get_local_id(0)+1,get_local_id(1)+1,get_local_id(2)+1};
    unsigned int ldx = ll[0]*bl[0]+ll[1]*bl[1]+ll[2]*bl[2];
    
    uchar i,j;
    
    Xl[ldx] = X[idx];
    
    for (i = 0; i < 3; i++) {
        if ((ll[i] == 1) && (ii[i] != 0)) Xl[ldx-bl[i]] = X[idx-bg[i]];
        if ((ll[i] == NL-2) && (ii[i] != ng[i]-1)) Xl[ldx+bl[i]] = X[idx+bg[i]];
        for (j = 0; j < i; j++) {
            if ((ll[i] == 1) && (ll[j] == 1) && (ii[i] != 0) && (ii[j] != 0)) Xl[ldx-bl[i]-bl[j]] = X[idx-bg[i]-bg[j]];
            if ((ll[i] == 1) && (ll[j] == NL-2) && (ii[i] != 0) && (ii[j] != ng[j]-1)) Xl[ldx-bl[i]+bl[j]] = X[idx-bg[i]+bg[j]];
            if ((ll[i] == NL-2) && (ll[j] == 1) && (ii[i] != ng[i]-1) && (ii[j] != 0)) Xl[ldx+bl[i]-bl[j]] = X[idx+bg[i]-bg[j]];
            if ((ll[i] == NL-2) && (ll[j] == NL-2) && (ii[i] != ng[i]-1) && (ii[j] != ng[j]-1)) Xl[ldx+bl[i]+bl[j]] = X[idx+bg[i]+bg[j]];
        }
    }
    
    if ((ll[0] == 1) && (ll[1] == 1) && (ll[2] == 1) && (ii[0] != 0) && (ii[1] != 0) && (ii[2] != 0))
        Xl[ldx-bl[0]-bl[1]-bl[2]] = X[idx-bg[0]-bg[1]-bg[2]];
    if ((ll[0] == 1) && (ll[1] == 1) && (ll[2] == NL-2) && (ii[0] != 0) && (ii[1] != 0) && (ii[2] != ng[2]-1))
        Xl[ldx-bl[0]-bl[1]+bl[2]] = X[idx-bg[0]-bg[1]+bg[2]];
    if ((ll[0] == 1) && (ll[1] == NL-2) && (ll[2] == 1) && (ii[0] != 0) && (ii[1] != ng[1]-1) && (ii[2] != 0))
        Xl[ldx-bl[0]+bl[1]-bl[2]] = X[idx-bg[0]+bg[1]-bg[2]];
    if ((ll[0] == NL-2) && (ll[1] == 1) && (ll[2] == 1) && (ii[0] != ng[0]-1) && (ii[1] != 0) && (ii[2] != 0))
        Xl[ldx+bl[0]-bl[1]-bl[2]] = X[idx+bg[0]-bg[1]-bg[2]];
    if ((ll[0] == 1) && (ll[1] == NL-2) && (ll[2] == NL-2) && (ii[0] != 0) && (ii[1] != ng[0]-1) && (ii[2] != ng[2]-1))
        Xl[ldx-bl[0]+bl[1]+bl[2]] = X[idx-bg[0]+bg[1]+bg[2]];
    if ((ll[0] == NL-2) && (ll[1] == 1) && (ll[2] == NL-2) && (ii[0] != ng[0]-1) && (ii[1] != 0) && (ii[2] != ng[2]-1))
        Xl[ldx+bl[0]-bl[1]+bl[2]] = X[idx+bg[0]-bg[1]+bg[2]];
    if ((ll[0] == NL-2) && (ll[1] == NL-2) && (ll[2] == 1) && (ii[0] != ng[0]-1) && (ii[1] != ng[1]-1) && (ii[2] != 0))
        Xl[ldx+bl[0]+bl[1]-bl[2]] = X[idx+bg[0]+bg[1]-bg[2]];
    if ((ll[0] == NL-2) && (ll[1] == NL-2) && (ll[2] == NL-2) && (ii[0] != ng[0]-1) && (ii[1] != ng[1]-1) && (ii[2] != ng[2]-1))
        Xl[ldx+bl[0]+bl[1]+bl[2]] = X[idx+bg[0]+bg[1]+bg[2]];
    
    barrier(CLK_LOCAL_MEM_FENCE);   
}

void __attribute__((overloadable)) 
toLocal(__global float3* X, __local float3* Xl)
{      
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    unsigned int ll[3] = {get_local_id(0)+1,get_local_id(1)+1,get_local_id(2)+1};
    unsigned int ldx = ll[0]*bl[0]+ll[1]*bl[1]+ll[2]*bl[2];
    
    uchar i,j;
    
    Xl[ldx] = X[idx];
    
    for (i = 0; i < 3; i++) {
        if ((ll[i] == 1) && (ii[i] != 0)) Xl[ldx-bl[i]] = X[idx-bg[i]];
        if ((ll[i] == NL-2) && (ii[i] != ng[i]-1)) Xl[ldx+bl[i]] = X[idx+bg[i]];
        for (j = 0; j < i; j++) {
            if ((ll[i] == 1) && (ll[j] == 1) && (ii[i] != 0) && (ii[j] != 0)) Xl[ldx-bl[i]-bl[j]] = X[idx-bg[i]-bg[j]];
            if ((ll[i] == 1) && (ll[j] == NL-2) && (ii[i] != 0) && (ii[j] != ng[j]-1)) Xl[ldx-bl[i]+bl[j]] = X[idx-bg[i]+bg[j]];
            if ((ll[i] == NL-2) && (ll[j] == 1) && (ii[i] != ng[i]-1) && (ii[j] != 0)) Xl[ldx+bl[i]-bl[j]] = X[idx+bg[i]-bg[j]];
            if ((ll[i] == NL-2) && (ll[j] == NL-2) && (ii[i] != ng[i]-1) && (ii[j] != ng[j]-1)) Xl[ldx+bl[i]+bl[j]] = X[idx+bg[i]+bg[j]];
        }
    }
    
    if ((ll[0] == 1) && (ll[1] == 1) && (ll[2] == 1) && (ii[0] != 0) && (ii[1] != 0) && (ii[2] != 0))
        Xl[ldx-bl[0]-bl[1]-bl[2]] = X[idx-bg[0]-bg[1]-bg[2]];
    if ((ll[0] == 1) && (ll[1] == 1) && (ll[2] == NL-2) && (ii[0] != 0) && (ii[1] != 0) && (ii[2] != ng[2]-1))
        Xl[ldx-bl[0]-bl[1]+bl[2]] = X[idx-bg[0]-bg[1]+bg[2]];
    if ((ll[0] == 1) && (ll[1] == NL-2) && (ll[2] == 1) && (ii[0] != 0) && (ii[1] != ng[1]-1) && (ii[2] != 0))
        Xl[ldx-bl[0]+bl[1]-bl[2]] = X[idx-bg[0]+bg[1]-bg[2]];
    if ((ll[0] == NL-2) && (ll[1] == 1) && (ll[2] == 1) && (ii[0] != ng[0]-1) && (ii[1] != 0) && (ii[2] != 0))
        Xl[ldx+bl[0]-bl[1]-bl[2]] = X[idx+bg[0]-bg[1]-bg[2]];
    if ((ll[0] == 1) && (ll[1] == NL-2) && (ll[2] == NL-2) && (ii[0] != 0) && (ii[1] != ng[0]-1) && (ii[2] != ng[2]-1))
        Xl[ldx-bl[0]+bl[1]+bl[2]] = X[idx-bg[0]+bg[1]+bg[2]];
    if ((ll[0] == NL-2) && (ll[1] == 1) && (ll[2] == NL-2) && (ii[0] != ng[0]-1) && (ii[1] != 0) && (ii[2] != ng[2]-1))
        Xl[ldx+bl[0]-bl[1]+bl[2]] = X[idx+bg[0]-bg[1]+bg[2]];
    if ((ll[0] == NL-2) && (ll[1] == NL-2) && (ll[2] == 1) && (ii[0] != ng[0]-1) && (ii[1] != ng[1]-1) && (ii[2] != 0))
        Xl[ldx+bl[0]+bl[1]-bl[2]] = X[idx+bg[0]+bg[1]-bg[2]];
    if ((ll[0] == NL-2) && (ll[1] == NL-2) && (ll[2] == NL-2) && (ii[0] != ng[0]-1) && (ii[1] != ng[1]-1) && (ii[2] != ng[2]-1))
        Xl[ldx+bl[0]+bl[1]+bl[2]] = X[idx+bg[0]+bg[1]+bg[2]];
    
    barrier(CLK_LOCAL_MEM_FENCE);   
}






void __attribute__((overloadable)) 
toPrivate(__global float3* X, float3* Xp)
{      
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    unsigned int jj[3] = {1,1,1};
    unsigned int jdx = jj[0]*bp[0]+jj[1]*bp[1]+jj[2]*bp[2];
    
    char i,j,k;

    for (i = -1; i < 2; i++) 
        for (j = -1; j < 2; j++) 
            for (k = -1; k < 2; k++)  
                if ((ii[0]+i>=0) && (ii[0]+i<=ng[0]-1) && (ii[1]+j>=0) && (ii[1]+j<=ng[1]-1) && (ii[2]+k>=0) && (ii[2]+k<=ng[2]-1)) 
                    Xp[jdx + i*bp[0] + j*bp[1] + k*bp[2]] = X[idx + i*bg[0] + j*bg[1] + k*bg[2]]; else 
                        Xp[jdx + i*bp[0] + j*bp[1] + k*bp[2]] = (float3)(0);
    
            
}






__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE)))
void Jacobian(__global float3* X, __global float3* J)
{
    __local float3 Xl[NL*NL*NL];
    
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    unsigned int ll[3] = {get_local_id(0)+1,get_local_id(1)+1,get_local_id(2)+1};
    unsigned int ldx = ll[0]*bl[0]+ll[1]*bl[1]+ll[2]*bl[2];
    
    uchar i;
    toLocal(X,Xl);
    
    for (i = 0; i < 3; i++) J[idx + i*NX*NY*NZ] = Deriv(Xl, ldx, i, 1);
    barrier(CLK_GLOBAL_MEM_FENCE);
}



struct q_str prep(__global float3* X, __global float3* V, __global float* T, __global float3* B, __global float3* dB)
{
    __local float3 Xl[NL*NL*NL];
    __local float3 dXl[NL*NL*NL];
    //__local float Tl[NL*NL*NL];
    
    
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    unsigned int ll[3] = {get_local_id(0)+1,get_local_id(1)+1,get_local_id(2)+1};
    unsigned int ldx = ll[0]*bl[0]+ll[1]*bl[1]+ll[2]*bl[2];
    
    unsigned int jj[3] = {1,1,1};
    unsigned int jdx = jj[0]*bp[0]+jj[1]*bp[1]+jj[2]*bp[2];
    
    struct q_str Q;
    
    uchar i,j;
    //float3 Xp[27];
    
    toLocal(X,Xl);
    //toPrivate(X,Xp);
    
    //Xl[ldx] = Xp[jdx];
    
    //loading field to private memory
    Q.B = B[idx];
    
    
    for (i = 0; i < 3; i++) {
        Q.dB[i] = dB[idx + i*NX*NY*NZ];
        
        //computing first and second derivatives of X
        Q.d2X[mm(i,i)] = Deriv(Xl, ldx, i, 2);
        Q.dX[i] = Deriv(Xl, ldx, i, 1);
        barrier(CLK_LOCAL_MEM_FENCE);
        
        //loading Jacobian to local memory
        if (i != 0) dXl[ldx] = Q.dX[i];
        
        //computing cross-dimentional secong derivatives of X
        
        for (j = 0; j < i; j++) {
            if (ll[j] == 1) dXl[ldx-bl[j]] = Deriv(Xl, ldx-bl[j], i, 1);
            if (ll[j] == NL-2) dXl[ldx+bl[j]] = Deriv(Xl, ldx+bl[j], i, 1);
            barrier(CLK_LOCAL_MEM_FENCE);
            Q.d2X[mm(i,j)] = Deriv(dXl, ldx, j, 1);
        }
        
        
        
    }
    
    //jacobian(Xp, Q.dX);
    //hessian(Xp, Q.d2X);
    /*barrier(CLK_LOCAL_MEM_FENCE);
    toLocal(T,Tl);

    for (i = 0; i < 3; i++) {
        Q.dT[i] = Deriv(Tl, ldx, i, 1);
    }
    */
    
    
    Q.dX[0].x += 1;
    Q.dX[1].y += 1;
    Q.dX[2].z += 1;
    
    //-----------------------------the rest of code uses only private memory-------------------------------------
    
    Q.det_dX = Q.dX[0].x*Q.dX[1].y*Q.dX[2].z + Q.dX[0].y*Q.dX[1].z*Q.dX[2].x + Q.dX[0].z*Q.dX[1].x*Q.dX[2].y - 
           Q.dX[0].x*Q.dX[1].z*Q.dX[2].y - Q.dX[0].y*Q.dX[1].x*Q.dX[2].z - Q.dX[0].z*Q.dX[1].y*Q.dX[2].x;
    
    Q.dXb[0] = Cross(Q.dX[1], Q.dX[2])/(float3)(Q.det_dX);
    Q.dXb[1] = Cross(Q.dX[2], Q.dX[0])/(float3)(Q.det_dX);
    Q.dXb[2] = Cross(Q.dX[0], Q.dX[1])/(float3)(Q.det_dX);
    
    Q.Bx.x = Q.dX[0].x*Q.B.x + Q.dX[1].x*Q.B.y + Q.dX[2].x*Q.B.z;
    Q.Bx.y = Q.dX[0].y*Q.B.x + Q.dX[1].y*Q.B.y + Q.dX[2].y*Q.B.z;
    Q.Bx.z = Q.dX[0].z*Q.B.x + Q.dX[1].z*Q.B.y + Q.dX[2].z*Q.B.z;
    
    Q.Bx /= (float3)Q.det_dX;
    
    for (i = 0; i < 3; i++) {
        Q.d_det_dX[i] = 0.;   
        for (j = 0; j < 3; j++) Q.d_det_dX[i] += Q.d2X[mm(i,j)].x*Q.dXb[j].x + Q.d2X[mm(i,j)].y*Q.dXb[j].y + Q.d2X[mm(i,j)].z*Q.dXb[j].z;
        Q.d_det_dX[i] *= Q.det_dX;
    }
       
    for (i = 0; i < 3; i++) {
        Q.dBx[i] = -(float3)(Q.d_det_dX[i])*Q.Bx;

        Q.dBx[i].x += Q.dX[0].x*Q.dB[i].x + Q.dX[1].x*Q.dB[i].y + Q.dX[2].x*Q.dB[i].z;
        Q.dBx[i].y += Q.dX[0].y*Q.dB[i].x + Q.dX[1].y*Q.dB[i].y + Q.dX[2].y*Q.dB[i].z;
        Q.dBx[i].z += Q.dX[0].z*Q.dB[i].x + Q.dX[1].z*Q.dB[i].y + Q.dX[2].z*Q.dB[i].z;
        
        Q.dBx[i].x += Q.d2X[mm(i,0)].x*Q.B.x + Q.d2X[mm(i,1)].x*Q.B.y + Q.d2X[mm(i,2)].x*Q.B.z;
        Q.dBx[i].y += Q.d2X[mm(i,0)].y*Q.B.x + Q.d2X[mm(i,1)].y*Q.B.y + Q.d2X[mm(i,2)].y*Q.B.z;
        Q.dBx[i].z += Q.d2X[mm(i,0)].z*Q.B.x + Q.d2X[mm(i,1)].z*Q.B.y + Q.d2X[mm(i,2)].z*Q.B.z;

        Q.dBx[i] /= (float3)(Q.det_dX);
    }
    return Q;
}


float3 Lorenz(struct q_str Q) {
    
    float3 out;
    float3 current = (float3)0;
    
    uchar i;
    
    for (i = 0; i < 3; i++) {
        current += Cross(Q.dXb[i], Q.dBx[i]);
    }
    out = Cross(current, Q.Bx)*Q.det_dX;
    return out;
}

float3 F1(struct q_str Q) {    
    float3 out;
    out.x = Q.d_det_dX[0]/Q.det_dX;
    out.y = Q.d_det_dX[1]/Q.det_dX;
    out.z = Q.d_det_dX[2]/Q.det_dX;
    return out;
}

__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE)))
void Step(__global float3* X, __global float3* Fx, __global float3* V, __global float3* Fv, __global float* T, __global float3* Ft,
          __global float3* B, __global float3* dB,
               float step)
{       
    //__local float3 Xl[NL*NL*NL];
    //__local float3 Vl[NL*NL*NL];
    //__local float3 Tl[NL*NL*NL];
    
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    unsigned int ll[3] = {get_local_id(0)+1,get_local_id(1)+1,get_local_id(2)+1};
    unsigned int ldx = ll[0]*bl[0]+ll[1]*bl[1]+ll[2]*bl[2];

    //toLocal(X,Xl);
    //toLocal(V,Vl);
    //toLocal(T,Tl);
    
    struct q_str Q;
    Q = prep(X, V, T, B, dB);
    
    Fx[idx] = V[idx];
    Fv[idx] = Lorenz(Q);
    barrier(CLK_GLOBAL_MEM_FENCE);
}

