// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <map>
#include <set>
#include <bits/stdc++.h>
using namespace std;

#define	WIDTH			256
#define HEIGHT			256
//	#define	TOTAL_PIXEL		65025	// 255x255 (WIDTH*HEIGHT)
#define ANGLE			10		// for dest
#define SRC_ANGLE		10
#define CAMNUM			10
#define CAMNUM_2		20

#define CENTER_X		128		// WIN_WIDTH/2
#define CENTER_Y		128		// WIN_HEIGHT/2
#define ART_ANGULAR		12
#define ART_RADIAL 		3
#define ART_COEF 		35//36
#define ART_COEF_q4 	18
#define	ART_LUT_RADIUS	50		// Zernike basis function radius
#define	ART_LUT_SIZE	101		// (ART_LUT_RADIUS*2+1)
#define PI				3.141592653
#define HYPOT(x,y)		sqrt((x)*(x)+(y)*(y))
#define	FD_COEF	10

// #2
#define	SPHERE_NUM	64
#define HAMONIC_NUM	64
#define	SH_KEY_NUM	SPHERE_NUM * HAMONIC_NUM

// #3
#define S3D_KEY_NUM		22

extern "C" void run(unsigned char * q8_table, unsigned char * align10, char * destfn,
                unsigned char * dest_ArtCoeff, unsigned char * dest_FdCoeff_q8,
                unsigned char * dest_CirCoeff_q8, unsigned char * dest_EccCoeff_q8)
                {

	// for region shape descriptor
	FILE			*fpt;

	// initialize: read camera pair
	fpt = fopen("./load_data/q8_table", "rb");
	fread(q8_table, sizeof(unsigned char), 65536, fpt);
	fclose(fpt);

	// initialize: read camera pair
	fpt = fopen("./load_data/align10.txt", "rb");
	fread(align10, sizeof(unsigned char), 60*CAMNUM_2, fpt);
	fclose(fpt);


    char filename[1000];

    sprintf(filename, "%s_q8_v1.8.art", destfn);
    if( (fpt = fopen(filename, "rb")) == NULL )
        {	printf("%s does not exist.\n", filename); return;	}
    fread(dest_ArtCoeff, ANGLE * CAMNUM * ART_COEF, sizeof(unsigned char), fpt);
    fclose(fpt);

    sprintf(filename, "%s_q8_v1.8.fd", destfn);
    if( (fpt = fopen(filename, "rb")) == NULL )
        {	printf("%s does not exist.\n", filename); return;}
    fread(dest_FdCoeff_q8, sizeof(unsigned char),  ANGLE * CAMNUM * FD_COEF, fpt);
    fclose(fpt);

    sprintf(filename, "%s_q8_v1.8.cir", destfn);
    if( (fpt = fopen(filename, "rb")) == NULL )
        {	printf("%s does not exist.\n", filename);	return;	}
    fread(dest_CirCoeff_q8, sizeof(unsigned char),  ANGLE * CAMNUM, fpt);
    fclose(fpt);

    sprintf(filename, "%s_q8_v1.8.ecc", destfn);
    if( (fpt = fopen(filename, "rb")) == NULL )
        {	printf("%s does not exist.\n", filename);	return;	}
    fread(dest_EccCoeff_q8, sizeof(unsigned char),  ANGLE * CAMNUM, fpt);
    fclose(fpt);

}
