#ifndef _DEBUG
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
template<class T> void _R(T& x) { cin >> x; }
void _R(int& x) { scanf("%d", &x); }
void _R(int64_t& x) { scanf("%lld", &x); }
void _R(double& x) { scanf("%lf", &x); }
void _R(char& x) { scanf(" %c", &x); }
void _R(char* x) { scanf("%s", x); }
void read() {}
template<class T, class... U> void read(T&& head, U &&... tail) { _R(head); read(tail...); }
template<class T> void _W(const T& x) { cout << x; }
void _W(const int& x) { printf("%d", x); }
void _W(const int64_t& x) { printf("%lld", x); }
void _W(const double& x) { printf("%.16f", x); }
void _W(const char& x) { putchar(x); }
void _W(const char* x) { printf("%s", x); }
template<class T, class U> void _W(const pair<T, U>& x) { _W("{"); _W(x.first); putchar(','); _W(x.second); _W("}"); }
template<class T> void _W(const vector<T>& x) { for (auto i = x.begin(); i != x.end(); _W(*i++)) if (i != x.cbegin()) putchar(' '); }
void print() { }
template<class T, class... U> void print(const T& head, const U &... tail) { _W(head); putchar(sizeof...(tail) ? ' ' : '\n'); print(tail...); }
#else

#include "types.h"
#include "oilib2.h"
using namespace lib2::p138;

#endif

int main()
{
//#ifdef _DEBUG
//    freopen("1.in", "r", stdin);
//    //freopen("tmp.out","w",stdout);
//#endif

    read_case();
    solve();

    return 0;
}