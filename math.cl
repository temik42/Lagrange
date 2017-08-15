#define mm(i,j) (i>j)?i*(i+1)/2+j:j*(j+1)/2+i //"smart" index for triangular matrices
#define PI 3.14159265
    
__constant uchar bp[3] = {9,3,1};
__constant uchar jdx = 13;

float Dot(float3 U, float3 V)
{    
    return U.x*V.x+U.y*V.y+U.z*V.z;
}

float3 Cross(float3 U, float3 V)
{
    float3 Y;
    Y.x = U.y*V.z - U.z*V.y;
    Y.y = U.z*V.x - U.x*V.z;
    Y.z = U.x*V.y - U.y*V.x;
    
    return Y;
}

void __attribute__((overloadable)) 
jacobian(float* Xp, float* dX)
{  
    uchar i;
    
    for (i = 0; i < 3; i++) 
        dX[i] = (Xp[jdx+bp[i]] - Xp[jdx-bp[i]])*0.5f;       
}

void __attribute__((overloadable)) 
jacobian(float3* Xp, float3* dX)
{  
    uchar i;
    
    for (i = 0; i < 3; i++) 
        dX[i] = (Xp[jdx+bp[i]] - Xp[jdx-bp[i]])*0.5f;       
}


void __attribute__((overloadable)) 
hessian(float* Xp, float* d2X)
{  
    uchar i,j;
    
    for (i = 0; i < 3; i++) {
        d2X[mm(i,i)] = Xp[jdx+bp[i]] + Xp[jdx-bp[i]] - Xp[jdx]*2.0f;
        for (j = 0; j < i; j++) {
            d2X[mm(i,j)] = (Xp[jdx+bp[i]+bp[j]] + Xp[jdx-bp[i]-bp[j]] - Xp[jdx+bp[i]-bp[j]] - Xp[jdx-bp[i]+bp[j]])*0.25f;
        }
    }
}


void __attribute__((overloadable)) 
hessian(float3* Xp, float3* d2X)
{   
    uchar i,j;
    
    for (i = 0; i < 3; i++) {
        d2X[mm(i,i)] = Xp[jdx+bp[i]] + Xp[jdx-bp[i]] - Xp[jdx]*2.0f;
        for (j = 0; j < i; j++) {
            d2X[mm(i,j)] = (Xp[jdx+bp[i]+bp[j]] + Xp[jdx-bp[i]-bp[j]] - Xp[jdx+bp[i]-bp[j]] - Xp[jdx-bp[i]+bp[j]])*0.25f;
        }
    }
}



