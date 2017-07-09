#pragma once
#include <Windows.h>

#define CheckPerf(call)                                                           \
{                                                                                 \
	LARGE_INTEGER t1, t2, tc;                                                     \
	QueryPerformanceFrequency(&tc);                                               \
	QueryPerformanceCounter(&t1);                                                 \
	call;                                                                         \
	QueryPerformanceCounter(&t2);                                                 \
	printf("Use Time:%f\n", (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart);       \
};