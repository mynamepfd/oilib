#pragma once

#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<algorithm>
#include<string>
#include<iostream>
#include<sstream>
#include<vector>
#include<list>
#include<stack>
#include<queue>
#include<set>
#include<map>
#include<bitset>
#include<limits.h>
#include<assert.h>
using namespace std;
#define all(X) (X).begin(), (X).end()
#define rep(I,A,B) for(int I=(A);I<(B);I++)
#define repd(I,A,B) for(int I=(A);I>(B);I--)
#define sort_unique(c) (sort(c.begin(),c.end()), c.resize(distance(c.begin(),unique(c.begin(),c.end()))))
typedef long long LL;
typedef pair<int, int> PII;
typedef pair<LL, LL> PLL;
typedef vector<int> VI;
typedef vector<PII> VPII;
typedef vector<LL> VLL;
typedef vector<PLL> VPLL;
