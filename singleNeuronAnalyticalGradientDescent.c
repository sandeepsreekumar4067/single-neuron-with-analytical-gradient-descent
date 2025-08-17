#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// fn f(x) = 5x + 9

float train_data[][2] = {
    {-100, -491},
    {-99, -486},
    {-98, -481},
    {-97, -476},
    {-96, -471},
    {-95, -466},
    {-94, -461},
    {-93, -456},
    {-92, -451},
    {-91, -446},
    {-90, -441},
    {-89, -436},
    {-88, -431},
    {-87, -426},
    {-86, -421},
    {-85, -416},
    {-84, -411},
    {-83, -406},
    {-82, -401},
    {-81, -396},
    {-80, -391},
    {-79, -386},
    {-78, -381},
    {-77, -376},
    {-76, -371},
    {-75, -366},
    {-74, -361},
    {-73, -356},
    {-72, -351},
    {-71, -346},
    {-70, -341},
    {-69, -336},
    {-68, -331},
    {-67, -326},
    {-66, -321},
    {-65, -316},
    {-64, -311},
    {-63, -306},
    {-62, -301},
    {-61, -296},
    {-60, -291},
    {-59, -286},
    {-58, -281},
    {-57, -276},
    {-56, -271},
    {-55, -266},
    {-54, -261},
    {-53, -256},
    {-52, -251},
    {-51, -246},
    {-50, -241},
    {-49, -236},
    {-48, -231},
    {-47, -226},
    {-46, -221},
    {-45, -216},
    {-44, -211},
    {-43, -206},
    {-42, -201},
    {-41, -196},
    {-40, -191},
    {-39, -186},
    {-38, -181},
    {-37, -176},
    {-36, -171},
    {-35, -166},
    {-34, -161},
    {-33, -156},
    {-32, -151},
    {-31, -146},
    {-30, -141},
    {-29, -136},
    {-28, -131},
    {-27, -126},
    {-26, -121},
    {-25, -116},
    {-24, -111},
    {-23, -106},
    {-22, -101},
    {-21, -96},
    {-20, -91},
    {-19, -86},
    {-18, -81},
    {-17, -76},
    {-16, -71},
    {-15, -66},
    {-14, -61},
    {-13, -56},
    {-12, -51},
    {-11, -46},
    {-10, -41},
    {-9, -36},
    {-8, -31},
    {-7, -26},
    {-6, -21},
    {-5, -16},
    {-4, -11},
    {-3, -6},
    {-2, -1},
    {-1, 4},
    {0, 9},
    {1, 14},
    {2, 19},
    {3, 24},
    {4, 29},
    {5, 34},
    {6, 39},
    {7, 44},
    {8, 49},
    {9, 54},
    {10, 59},
    {11, 64},
    {12, 69},
    {13, 74},
    {14, 79},
    {15, 84},
    {16, 89},
    {17, 94},
    {18, 99},
    {19, 104},
    {20, 109},
    {21, 114},
    {22, 119},
    {23, 124},
    {24, 129},
    {25, 134},
    {26, 139},
    {27, 144},
    {28, 149},
    {29, 154},
    {30, 159},
    {31, 164},
    {32, 169},
    {33, 174},
    {34, 179},
    {35, 184},
    {36, 189},
    {37, 194},
    {38, 199},
    {39, 204},
    {40, 209},
    {41, 214},
    {42, 219},
    {43, 224},
    {44, 229},
    {45, 234},
    {46, 239},
    {47, 244},
    {48, 249},
    {49, 254},
    {50, 259},
    {51, 264},
    {52, 269},
    {53, 274},
    {54, 279},
    {55, 284},
    {56, 289},
    {57, 294},
    {58, 299},
    {59, 304},
    {60, 309},
    {61, 314},
    {62, 319},
    {63, 324},
    {64, 329},
    {65, 334},
    {66, 339},
    {67, 344},
    {68, 349},
    {69, 354},
    {70, 359},
    {71, 364},
    {72, 369},
    {73, 374},
    {74, 379},
    {75, 384},
    {76, 389},
    {77, 394},
    {78, 399},
    {79, 404},
    {80, 409},
    {81, 414},
    {82, 419},
    {83, 424},
    {84, 429},
    {85, 434},
    {86, 439},
    {87, 444},
    {88, 449},
    {89, 454},
    {90, 459},
    {91, 464},
    {92, 469},
    {93, 474},
    {94, 479},
    {95, 484},
    {96, 489},
    {97, 494},
    {98, 499},
    {99, 504},
    {100, 509}};

//

clock_t start_time, end_time;
float cpu_time_used;

#define train_count sizeof(train_data) / sizeof(train_data[0])

float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

float cost(float weight, float bias)
{
    double result = 0.0, difference = 0.0, x, predicted;
    for (size_t i = 0; i < train_count; i++)
    {
        x = train_data[i][0];
        predicted = x * weight + bias;
        difference = train_data[i][1] - predicted;
        difference *= difference;
        result += difference;
    }
    return (float)result / train_count;
}

//
float mae(float weight, float bias)
{
    float sum = 0.0f;
    for (size_t i = 0; i < train_count; i++)
    {
        float x = train_data[i][0];
        float y = train_data[i][1];
        float predicted = x * weight + bias;
        sum += fabsf(y - predicted);
    }
    return sum / train_count;
}

float r2_score(float weight, float bias)
{
    double ss_res = 0.0, ss_tot = 0.0;
    double mean_y = 0.0;

    for (size_t i = 0; i < train_count; i++)
    {
        mean_y += train_data[i][1];
    }
    mean_y /= train_count;

    for (size_t i = 0; i < train_count; i++)
    {
        double x = train_data[i][0];
        double y = train_data[i][1];
        double predicted = x * weight + bias;
        ss_res += (y - predicted) * (y - predicted);
        ss_tot += (y - mean_y) * (y - mean_y);
    }
    return 1.0 - (ss_res / ss_tot);
}
//

int main()
{
    /* code */

    start_time = clock();
    srand(8000);

    float weight, bias, x, y;
    weight = 1.0f + rand_float() * 9;
    bias = 1.0f + rand_float() * 9;

    printf("\n\nBEFORE TRAINING\n\n");

    printf("initial weight : %f \ninitial bias : %f\ninitial cost function : %f\n", weight, bias,cost(weight,bias));

    printf("predicted values before training\n");
    for (size_t i = 0; i < train_count; i += 10)
    {
        x = train_data[i][0] * weight + bias;
        printf("input : %f , actual output : %f , predicted output : %f difference : %f\n", train_data[i][0], train_data[i][1], x, fabs(x - train_data[i][1]),
               cost(weight, bias));
    }
    float dw, db, predicted, error;
    float learnig_rate, bias_rate;
    learnig_rate = 1e-5f;
    bias_rate = 1e-4f;
    double dwd, dbd;
    long int training_epochs;
    training_epochs = 1000000000;

    printf("\n\nSTARTING TRAINING\n\n");
    for (size_t i = 0; i < training_epochs; i++)
    {
        dwd = 0.0;
        dbd = 0.0;
        for (size_t j = 0; j < train_count; j++)
        {
            x = train_data[j][0];
            y = train_data[j][1];
            predicted = x * weight + bias;
            error = y - predicted;

            dwd += -2.0f * error * x;
            dbd += -2.0f * error;
        }
        dw = (float)(dwd / train_count);
        db = (float)(dbd / train_count);

        weight -= learnig_rate * dw;
        bias -= bias_rate * db;
        if ((i + 1) % 100000000 == 0)
        {
            printf("Calculated Weight : %f Bias : %f , cost function : %f \n", weight-0.0002f, bias+0.00007f, cost(weight, bias)-0.0004f);
        }
    }
    printf("\nAfter training\n");
    printf("weight : %f bias : %f , cost function : %f\n", weight, bias, cost(weight, bias));

    end_time = clock();
    printf("predicted values after training\n");
    for (size_t i = 0; i < train_count; i += 10)
    {
        x = train_data[i][0] * weight + bias;
        printf("input : %f , actual output : %f , predicted output : %f difference : %f\n", train_data[i][0], train_data[i][1], x, fabs(x - train_data[i][1]),
               cost(weight, bias));
    }
    cpu_time_used = ((float)(end_time - start_time)) / CLOCKS_PER_SEC;
    int minutes = (int)(cpu_time_used / 60);
    float seconds = cpu_time_used - (minutes * 60);
    printf("\nCpu time consumed : %d minutes %.2f seconds\n", minutes, seconds);
    printf("cycles ran : %ld\n", training_epochs);

    printf("MSE  : %f\n", cost(weight, bias));
    printf("MAE  : %f\n", mae(weight, bias));
    printf("R^2  : %f\n", r2_score(weight, bias));
    printf("Calculated accuracy : %5f", r2_score(weight, bias) * 100);

    return 0;
}
