#include "types.h"
#include "oilib2.h"

namespace lib2 {

namespace p122 {

	const int MAX_N = 200001;
	const int MAX_M = 200001;

	int n, m; // 矿石总数、区间总数和标准值
	LL s;
	int w[MAX_N], v[MAX_N]; // 矿石重量和价值
	int l[MAX_M], r[MAX_M];

	void read_case() {
		read(n, m, s);
		rep(i, 1, n + 1)
			read(w[i], v[i]);
		rep(i, 1, m + 1)
			read(l[i], r[i]);
	}

	LL check(int W) {
		LL y = 0;

		VI cnt(n + 1); // vcnt[i]表示前i个重量超过W的矿石总数
		VLL vsum(n + 1); // vsum[i]表示前i个重量超过W的矿石的价值之和
		rep(i, 1, n + 1) {
			if (w[i] >= W) { cnt[i] = 1; vsum[i] = v[i]; }
			cnt[i] += cnt[i - 1];
			vsum[i] += vsum[i - 1];
		}
		rep(i, 1, m + 1) {
			y += 1LL * (cnt[r[i]] - cnt[l[i] - 1]) * (vsum[r[i]] - vsum[l[i] - 1]);
		}
		return y;
	}

	void solve() {
		int lb = 0, ub = 1000000;
		LL ans = -1;
		while (lb <= ub) {
			int W = (lb + ub) / 2;
			LL y = check(W);
			if (y > s) lb = W + 1;
			else ub = W - 1;
			if (ans == -1) ans = abs(s - y);
			else ans = min(ans, abs(s - y));
		}
		print(ans);
	}
}

namespace p123 {

	const int MAX_N = 50002;

	int L, n, m;
	int d[MAX_N];

	void read_case() {
		read(L, n, m);
		rep(i, 1, n + 1)
			read(d[i]);
		d[n + 1] = L;
	}

	// 能否移动至多M个岩石使得最短跳跃距离大于等于k
	int check(int k) {
		int i = 0, c = 0;
		while (i <= n) {
			int j = i + 1;
			while (j <= n + 1 && d[j] - d[i] < k) {
				j++;
				c++;
			}
			if (d[j] - d[i] < k) {
				return false;
			}
			i = j;
		}
		return c <= m;
	}

	const int INF = 1000000000;

	void solve() {
		// 目标是移走至多M个岩石使得最短跳跃距离尽可能地长
		int lb = 0, ub = INF;
		while (ub - lb > 1) { // [lb,ub)
			int mid = (lb + ub) / 2;
			if (check(mid)) lb = mid;
			else ub = mid;
		}
		cout << lb << endl;
	}
}

namespace p124 {

	const int MAXN = 1000009;
	const int MAXM = 1000009;

	int n, m, r[MAXN];
	int d[MAXM], s[MAXM], t[MAXM];

	void read_case() {
		read(n, m);
		rep(i, 0, n)
			read(r[i]);
		rep(i, 0, m)
			read(d[i], s[i], t[i]);
	}

	bool check(int mm)
	{
		VI a(n);
		rep(i, 0, mm + 1) {
			a[s[i]] += d[i];
			if (t[i] + 1 < n)
				a[t[i] + 1] -= d[i];
		}
		rep(i, 1, n)
			a[i] += a[i - 1];
		rep(i, 0, n)
			if (a[i] > r[i])
				return false;
		return true;
	}

	void solve() {
		rep(i, 0, m) {
			s[i] -= 1;
			t[i] -= 1;
		}
			
		int lo = 0;
		int hi = m - 1;
		int ans = -1;
		while (lo <= hi) {
			int mm = (lo + hi) / 2;
			if (check(mm))
				lo = mm + 1;
			else {
				ans = mm;
				hi = mm - 1;
			}
		}
		if (ans == -1)
			print(0);
		else {
			print(-1);
			print(ans + 1);
		}
	}
}

namespace p125 {

	const int MAX_N = 100001, MAX_M = 100001;

	int n;
	int h[MAX_N];
	int x0;
	int m;
	int s[MAX_M], x[MAX_M];

	void read_case() {
		read(n);
		rep(i, 1, n + 1)
			read(h[i]);
		read(x0);
		read(m);
		rep(i, 1, m + 1)
			read(s[i], x[i]);
	}

	vector<VLL> jmp, f, g; // jmp[k][i] 表示从i出发开2^k轮后所在城市
						   // f[k][i] 表示从i出发小a开2^k轮后的距离
						   // g[k][i] 表示从i出发小b开2^k轮后的距离

	// 求出从j出发总里程为x的情况下小a和小b的里程总数
	void query(int j, int x, LL& da, LL& db) {
		da = db = 0;
		repd(i, 21, -1) {
			if (f[i][j] + g[i][j] <= x) {
				x -= f[i][j] + g[i][j];
				da += f[i][j];
				db += g[i][j];
				j = jmp[i][j];
			}
		}
		if (f[0][j] <= x) {
			x -= f[0][j];
			da += f[0][j];
		}
	}

	void solve() {
		// 以下开一轮指a,b各开一次

		set<PII> S;
		VI a(n + 1), b(n + 1);

		repd(i, n, 0) { // 找到小a和小b的从i出发的目的地,需要倒着往S插入数据
			S.insert({ h[i],i });

			vector<pair<int, PII>> v; //{距离,{海拔高度,城市id}}
			set<PII>::iterator pre, succ; 
			pre = succ = S.lower_bound({ h[i],i });
			if (pre != S.begin()) { // 取出距i城最近的海拔比它低的两座城市
				v.push_back({ 0,*(--pre) });
				if (pre != S.begin())
					v.push_back({ 0,*(--pre) });
			}
			if (++succ != S.end()) { // 取出距i城最近的海拔比它高的两座城市
				v.push_back({ 0,*succ });
				if (++succ != S.end()) {
					v.push_back({ 0,*succ });
				}
			}
			rep(j, 0, v.size())
				v[j].first = abs(v[j].second.first - h[i]);
			sort(all(v));
			if (v.size() > 0)b[i] = v[0].second.second;
			if (v.size() > 1)a[i] = v[1].second.second;
		}

		
		jmp = f = g = vector<VLL>(22, VLL(n + 1));
		rep(i, 1, n + 1) {
			jmp[0][i] = b[a[i]]; // 计算从i开始开一轮到达的城市
			if (a[i]) f[0][i] = abs(h[a[i]] - h[i]); // 计算a开的距离
			if (b[a[i]] && a[i]) g[0][i] = abs(h[b[a[i]]] - h[a[i]]); // 计算b开的距离
		}

		// 计算开2**i轮后的状态
		rep(i, 1, 22) {
			rep(j, 1, n + 1) {
				jmp[i][j] = jmp[i - 1][jmp[i - 1][j]];
				f[i][j] = f[i - 1][j] + f[i - 1][jmp[i - 1][j]];
				g[i][j] = g[i - 1][j] + g[i - 1][jmp[i - 1][j]];
			}
		}

		// da_,db_用于找出答案
		LL da_ = 1, db_ = 0, i_, h_, da, db;
		rep(i, 1, n + 1) {
			query(i, x0, da, db);
			// 比较 da/db < da_/db_
			if (da * db_ < db * da_) {
				i_ = i;
				h_ = h[i];
				da_ = da;
				db_ = db;
			}
		}
		print(i_);
		rep(i, 1, m + 1) {
			query(s[i], x[i], da, db);
			print(da, db);
		}
	}
}

namespace p126 {

	const int MAXN = 100000;
	const int MAXM = 100000;
	const int MAXK = 100000;

	int n, m, k; // n:城市数 m:边数 k:敌城数
	int a[MAXK]; // 无效顶点
	int U[MAXM], V[MAXM];

	void read_case() {
		read(n, m, k);
		rep(i, 0, k)
			read(a[i]);
		rep(i, 0, m)
			read(U[i], V[i]);
	}

	VI valid;
	vector<VI> adj, adjv;
	vector<VI> cnt;

	VI S, S1;
	vector<VI> cnt_;

	void del(int i, double c) {
		queue<int> Q;
		Q.push(i);
		while (Q.size() > 0) {
			int u = Q.front(); Q.pop();
			S[u] = 0;
			for (int v : adjv[u]) {
				cnt_[v][0] -= 1;
				double t = (double)cnt_[v][0] / (double)cnt_[v][1];
				if (t < c && S[v] == 1)
					Q.push(v);
			}
		}
	}

	bool check(double c) {
		cnt_ = cnt;
		S = VI(n, 1);
		S1.clear();
		rep(i, 0, n) {
			if (valid[i] == 1)
				if (S[i] == 1) {
					double t = (double)cnt_[i][0] / (double)cnt_[i][1];
					if (t < c)
						del(i, c);
				}
			else
				S[i] = 0;
		}
		rep(i, 0, n)
			if (S[i] == 1)
				S1.push_back(i + 1);
		if (S1.size() > 0)
			return true;
		else
			return false;
	}

	void solve() {
		valid = VI(n, 1);
		rep(i, 0, k)
			valid[a[i] - 1] = 0;
		adj = vector<VI>(n);
		adjv = vector<VI>(n);
		rep(i, 0, m) {
			int u = U[i]-1;
			int v = V[i]-1;
			adj[u].push_back(v);
			adj[v].push_back(u);
			if (valid[v] == 1)
				adjv[u].push_back(v);
			if (valid[u] == 1)
				adjv[v].push_back(u);
		}
		cnt = vector<VI>(n, VI(2));
		rep(u, 0, n) {
			cnt[u][0] = adjv[u].size(); // 友方邻居数
			cnt[u][1] = adj[u].size(); // 总邻居数
		}
		
		double lo = 0, hi = 1;
		VI ans;
		rep(i, 0, 100) {
			double c = (lo + hi) / 2;
			if (check(c)) {
				ans = S1;
				lo = c;
			}
			else
				hi = c;
		}
		print(ans.size());
		print(ans);
	}
}

namespace p127 {
	const int MAXN = 300000;
	int n, a[MAXN];
	
	void read_case() {
		read(n);
		rep(i, 1, n+1)
			read(a[i]);
	}

	int gcd(int a, int b) {
		if (b == 0) return a;
		else return gcd(b, a % b);
	}

	vector<VI> f, g; // f[k][i]是从i开始长度为2^k的区间的最小值,同理g[k][i]是gcd

	// 检查是否存在长为mid的符合要求的区间
	int check(int mid) {
		rep(i, 1, n + 1) {
			int j = i + mid;
			if (j <= n + 1) {
				int k = log2(j - i);
				// [i,i+2^k]和[j-2^k,j]覆盖了从i开始长度为2^k的区间
				if (min(f[k][i], f[k][j - (1 << k)]) == gcd(g[k][i], g[k][j - (1 << k)]))
					return true;
			}
			else
				return false;

		}
		return false;
	}

	void solve() {
		
		f = g = vector<VI>(20, VI(n + 1));
		rep(i, 1, n + 1) {
			f[0][i] = a[i];
			g[0][i] = a[i];
		}
		//    W(0,':');
		//    W(f[0]);
		//    W(g[0]);

		rep(i, 1, 20) {
			int k = 1 << (i - 1);
			rep(j, 1, n + 1 - k) {
				f[i][j] = min(f[i - 1][j], f[i - 1][j + k]);
				g[i][j] = gcd(g[i - 1][j], g[i - 1][j + k]);
			}
			//    	W(i,':');
			//    	W(f[i]);
			//    	W(g[i]);
		}

		int lb = 0, ub = n + 1;
		while (ub - lb > 1) {
			int mid = (ub + lb) / 2;
			if (check(mid)) lb = mid;
			else ub = mid;
		}

		//W(lb);

		VI ans;
		rep(i, 1, n + 1) {
			int j = i + lb;
			if (j <= n + 1) {
				int k = log2(j - i);
				if (min(f[k][i], f[k][j - (1 << k)]) == gcd(g[k][i], g[k][j - (1 << k)]))
					ans.push_back(i);
			}
			else
				break;
		}
		print(ans.size(), lb - 1);
		print(ans);
	}
}

namespace p128 {

	const int MAXN = 100009;

	int n, a[MAXN];
	
	void read_case() {
		read(n);
		rep(i, 1, n + 1)
			read(a[i]);
	}
	
	LL sum[MAXN];
	struct POINT { LL x, y; };
	vector<POINT> points;

	// 定义一些工具函数
	bool cmpy(const int& a, const int& b) { return points[a].y < points[b].y; }
	LL dist(POINT* p1, POINT* p2) { return (p2->x - p1->x) * (p2->x - p1->x) + (p2->y - p1->y) * (p2->y - p1->y); }

	LL closest_pair(int ql, int qr) // 找到点ql到点qr中的最近点对距离
	{
		LL d, d1, d2;
		d = 1e18;
		if (ql == qr)
			return d;
		if (ql + 1 == qr)
			return dist(&points[ql], &points[qr]);
		int mid = (ql + qr) / 2;
		d1 = closest_pair(ql, mid);
		d2 = closest_pair(mid + 1, qr);
		d = min(d1, d2);
		VI tmp;
		for (int i = ql; i <= qr; i++)
			if (abs(points[i].x - points[mid].x) <= d)
				tmp.push_back(i);
		sort(all(tmp), cmpy);
		for (int i = 0; i < tmp.size(); i++)
			for (int j = i + 1; j <= min(i + 6, (int)tmp.size()-1); j++)
				d = min(d, dist(&points[tmp[i]], &points[tmp[j]]));
		return d;
	}

	void solve() {
		for (int i = 1; i <= n; i++) {
			sum[i] = sum[i - 1] + a[i];
			points.push_back({ 1ll * i, sum[i] });
		}
		print(closest_pair(1, n));
	}
}

namespace p129 {

	const int MAXN = 1 << 20, MAXM = 1000000;
	int n, a[MAXN];
	int m, q[MAXM];

	void read_case() {
		read(n);
		rep(i, 1, (1 << n)+1)
			read(a[i]);
		read(m);
		rep(i, 0, m)
			read(q[i]);
	}

	// f[k]表示每一个长为2^k的段内,左半部分与右半部分形成的逆序对的数量
	LL f[MAXN], g[MAXN]; 

	void merge_sort(int p, int r, int x)
	{
		if (p >= r)
			return;
		int q = (p + r) / 2;
		merge_sort(p, q, x - 1);
		merge_sort(q + 1, r, x - 1);
		for (int i = p; i <= q; i++)
		{
			f[x] += lower_bound(a + q + 1, a + r + 1, a[i]) - (a + q + 1); //找到右区间中比a[i]小的元素数量,a[i]与它们形成逆序对
			g[x] += (r - (q + 1) + 1) - (upper_bound(a + q + 1, a + r + 1, a[i]) - (a + q + 1)); // 右区间中一共有r-(q+1)+1个元素,找到右区间中小于等于a[i]的元素数量,a[i]与剩下的元素形成顺序对
		}
		sort(a + p, a + r + 1);
	}

	LL flip(int x)
	{
		LL ans = 0;
		for (int i = 1; i <= x; i++)
			swap(f[i], g[i]);
		for (int i = 1; i <= n; i++)
			ans += f[i];
		return ans;
	}

	void solve() {
		int sz = 1 << n;
		merge_sort(1, sz, n);
		for(int i=0; i<m; i++) {
			print(flip(q[i])); // 以2^k分段并翻转
		}
	}
}

namespace p1210 {

	const int INF = 1000000000;
	const int MAX_N = 100000;

	int N;
	LL K; // k表示从i出发走k条边
	int F[MAX_N], W[MAX_N]; // F[i]表示点i的出边指向的点,W[i]表示点i的出边的边权

	void read_case() {
		read(N, K);
		rep(i, 0, N)
			read(F[i]);
		rep(i, 0, N)
			read(W[i]);
	}

	int jmp[MAX_N][35];
	LL f[MAX_N][35];
	int g[MAX_N][35];

	void query(int u, LL &total, int &minv)
	{
		for (int i = 34; i >= 0; i--)
		{
			if ((K >> i) & 1) // 二进制分解
			{
				total += f[u][i];
				minv = min(minv, g[u][i]);
				u = jmp[u][i];
			}
		}
	}

	void solve() {
		
		for (int i = 0; i < N; i++)
			jmp[i][0] = F[i];

		for (int i = 0; i < N; i++)
			f[i][0] = g[i][0] = W[i];

		for (int j = 1; j <= 34; j++)
		{
			for (int i = 0; i < N; i++)
			{
				jmp[i][j] = jmp[jmp[i][j - 1]][j - 1];
				f[i][j] = f[i][j - 1] + f[jmp[i][j - 1]][j - 1];
				g[i][j] = min(g[i][j - 1], g[jmp[i][j - 1]][j - 1]);
			}
		}

		rep(i, 0, N) {
			LL total = 0;
			int minv = INF;
			query(i, total, minv);
			print(total, minv);
		}
	}
}

namespace p131 {
	const int MAXN = 100009;
	int n, a[MAXN];

	void read_case() {
		read(n);
		rep(i, 1, n + 1)
			read(a[i]);
	}

	void solve() {
		int ans = 0;
		for (int i = 1; i <= n; i++)
			if (a[i] > a[i - 1])
				ans += (a[i] - a[i - 1]);
		print(ans);
	}
}

namespace p132 {

	// 这是一个高精度*单精度的类
	struct BIGNUM {

		int v[5000];

		BIGNUM() {
			memset(v, 0, sizeof(v));
		}
		BIGNUM(int n) {
			memset(v, 0, sizeof(BIGNUM));
			for (; n; n /= 10)
				v[++v[0]] = n % 10;
		}

		void operator=(BIGNUM B)
		{
			memcpy(v, B.v, sizeof(v));
		}

		int& operator[](int index) {
			return v[index];
		}

		BIGNUM operator/(int B)
		{
			BIGNUM q;
			int k = 0;
			for (int i = v[0]; i >= 1; i--)
			{
				k = k * 10 + v[i];
				if (k >= B)
				{
					if (!q[0]) q[0] = i;
					q[i] = k / B;
					k %= B;
				}
			}
			return q;
		}

		BIGNUM operator*(int B)
		{
			BIGNUM c;
			int k = 0;
			for (int i = 1; i <= v[0]; i++)
			{
				c[i] += v[i] * B + k;
				c[i + 1] += c[i] / 10;
				c[i] %= 10;
			}
			for (c[0] = v[0]; c[c[0] + 1];)
			{
				c[0]++;
				c[c[0] + 1] += c[c[0]] / 10;
				c[c[0]] %= 10;
			}
			return c;
		}

		bool operator<=(BIGNUM& B) {
			if (v[0] > B[0])
				return false;
			else if (v[0] < B[0])
				return true;
			else
			{
				for (int i = v[0]; i >= 1; i--)
					if (v[i] > B[i])
						return false;
					else if (v[i] < B[i])
						return true;
				return true;
			}
		}
	};

	void print(BIGNUM& B) {
		for (int i = B[0]; i >= 1; i--)
			printf("%d", B[i]);
	}

	const int MAX_N = 1000;

	int n; // 大臣数量
	int A, B; //国王手中的数字
	int a[MAX_N], b[MAX_N]; // 大臣们手中的数字

	void read_case() {
		read(n);
		read(A,B);
		rep(i, 0, n)
			read(a[i], b[i]);
	}

	bool cmp(const VI& item0, const VI& item1) {
		return (item0[0] * item0[1]) < (item1[0] * item1[1]);
	}

	void solve() {
		vector<VI> seq;
		rep(i, 0, n) {
			VI  v = { a[i],b[i] };
			seq.push_back(v);
		}
		sort(all(seq), cmp);
		BIGNUM ans;
		BIGNUM K(A);
		rep(i, 0, n) {
			BIGNUM c = K / seq[i][1];
			if (ans <= c)
				ans = c;
			K = K * seq[i][0];
		}
		print(ans);
	}
}

namespace p133 {

	struct BIT {
		VI a;
		BIT(int n) { a = VI(n + 1); }
		int sum(int i) { int s = 0; while (i > 0) s += a[i], i -= (i & -i); return s; }
		void add(int i, int v) { while (i < a.size()) a[i] += v, i += (i & -i); }
	};

	const int MAX_N = 100000;
	int n, a[MAX_N], b[MAX_N];
	
	void read_case() {
		read(n);
		rep(i, 0, n) read(a[i]);
		rep(i, 0, n) read(b[i]);
	}

	void solve() {
		VI ord(n);
		VI a1, b1;
		rep(i, 0, n) a1.push_back(a[i]);
		sort(all(a1));
		rep(i, 0, n)
			ord[lower_bound(all(a1), a[i]) - a1.begin()] = i + 1;
		rep(i, 0, n) b1.push_back(b[i]);
		sort(all(b1));
		rep(i, 0, n)
			b[i] = ord[lower_bound(all(b1), b[i]) - b1.begin()];
		BIT bit = BIT(n);
		int ans = 0;
		repd(i, n - 1, -1) {
			ans = (ans + bit.sum(b[i] - 1)) % (100000000 - 3);
			bit.add(b[i], 1);
		}
		print(ans);
	}
}

namespace p134 {
	
	const int MAX_N = 1000;

	int n, m, k;
	int D[MAX_N];
	int t[MAX_N], a[MAX_N], b[MAX_N];

	void read_case() {
		read(n, m, k);
		rep(i, 1, n)
			read(D[i]);
		rep(i, 0, m)
			read(t[i], a[i], b[i]);
	}

	int MAXN;
	int c[MAX_N];
	int last[MAX_N], e[MAX_N];
	void solve() {
		MAXN = n + 1;
		rep(i, 0, m) {
			last[a[i]] = max(last[a[i]], t[i]);
			c[b[i]] += 1;
		}
		rep(i, 1, n)
			e[i + 1] = max(e[i], last[i]) + D[i];
		int ans = 0;
		rep(i, 0, m) {
			int tm = e[b[i]] - t[i];
			ans += tm;
		}
			
		rep(_, 0, k) {
			int max_save = 0;
			int this_i = 0;
			rep(i, 1, n) {
				int save = 0; //计算对D[i]使用加速器能节约的时间
				if (D[i] > 0)
					rep(j, i + 1, MAXN) { //从j开始抵达时间都会减一
					save += c[j];
					if (e[j] <= last[j]) //意味着从j + 1开始抵达时间不会变化
						break;
				}
				if (save > max_save) {
					max_save = save;
					this_i = i;
				}
			}
			ans -= max_save;
			D[this_i] -= 1;
			rep(i, this_i, n)
				e[i + 1] = max(e[i], last[i]) + D[i];
		}
		print(ans);
	}
}

namespace p135 {

	const int MAX_N = 100000, MAX_M = 100000;

	int n,m;
	int A[MAX_N], B[MAX_N];
	int C[MAX_M], D[MAX_M], K[MAX_M];

	void read_case()
	{
		read(n);
		rep(i, 0, n)
			read(A[i], B[i]);
		read(m);
		rep(i, 0, m)
			read(C[i], D[i], K[i]);
	}

	struct v4
	{
		int lo, hi, k, i;
	};
	bool cmp_lo(const v4& l, const v4& r) { return l.lo < r.lo; }

	v4 a[MAX_N]; // 待演奏的片段
	v4 b[MAX_M]; // 每个人的演奏能力
	int ans[MAX_N];

	void solve()
	{
		rep(i, 0, n) // 重新组织数据
			a[i] = { A[i], B[i], 0, i };
		rep(i, 0, m)
			b[i] = { C[i], D[i], K[i], i };
		sort(a, a + n, cmp_lo);
		sort(b, b + m, cmp_lo);

		bool found = true;

		set<pair<int, int> > s;
		int j = 0;
		for (int i = 0; i < n; i++)
		{
			while (j < m && b[j].lo <= a[i].lo)
			{
				s.insert(make_pair(b[j].hi, j));
				j += 1;
			}
			set<pair<int, int> >::iterator iter = s.lower_bound(make_pair(a[i].hi, 0)); //找到第一个能演奏该片段的人
			if (iter == s.end())
			{
				found = false;
				break;
			}
			int id = iter->second;
			ans[a[i].i] = b[id].i + 1; // 让他演奏
			b[id].k -= 1;
			if (b[id].k == 0)
				s.erase(iter);
		}

		if (found)
		{
			printf("YES\n");
			for (int i = 0; i < n; i++)
				printf("%d ", ans[i]);
		}
		else
		{
			printf("NO");
		}
	}
}

namespace p136 {

	const int MAXN = 100009;
	int n, m;
	int c[MAXN], w[MAXN];

	void read_case() {
		read(n, m);
		rep(i, 1, n + 1)
			read(c[i]);
		rep(i, 1, n + 1)
			read(w[i]);
	}

	struct cmp
	{
		bool operator()(int i, int j)
		{
			int a = w[i] * (100 - c[i] % 100);
			int b = w[j] * (100 - c[j] % 100);
			if (a <= b)
				return false;
			else
				return true;
		}
	};

	int ans[MAXN][2]; // ans[i][0]:在第i天花费的100元纸币数 ans[i][1]:花费的1元硬币数
	LL sum; // 愤怒值,要使用ll否则会溢出

	void solve() {
		priority_queue<int, std::vector<int>, cmp> q;

		for (int i = 1; i <= n; i++)
		{
			int k = c[i] % 100;
			ans[i][0] = c[i] / 100;
			ans[i][1] = k;
			m -= k;
			if (k > 0)
			{
				q.push(i);
				if (m < 0)
				{
					int day = q.top(); q.pop();
					m += 100;
					ans[day][0]++;
					ans[day][1] = 0;
					sum += w[day] * (100 - c[day] % 100);
				}
			}
		}
		printf("%lld\n", sum);
		for (int i = 1; i <= n; i++)
			printf("%d %d\n", ans[i][0], ans[i][1]);
	}
}

namespace p137 {
	const int MAX_N = 100000, MAX_M = 100000;

	int n, m;
	int a[MAX_N], b[MAX_M], p[MAX_M];

	void read_case() {
		read(n, m);
		rep(i, 0, n) read(a[i]);
		rep(i, 0, m) read(b[i]);
		rep(i, 0, m) read(p[i]);
	}

	void solve() {
		vector<VI> l;
		rep(i, 0, m)
			l.push_back({ b[i], p[i] });
		sort(all(l));
		priority_queue<int, vector<int>, greater<int> > Q;
		int count = 0;
		int j = 0;
		repd(i, n - 1, -1) {
			while (j < m && l[j][0] <= a[i]) {
				Q.push(l[j][1]);
				j += 1;
			}
			int cost = 0;
			while (Q.size() > 0) {
				if (cost + Q.top() <= a[i]) {
					cost += Q.top();
					Q.pop();
					count += 1;
				}
				else
					break;
			}
			int rem = a[i] - cost;
			if (Q.size() > 0) {
				int v = Q.top();
				v -= rem;
				Q.push(v);
			}
		}
		print(count);
	}
}

namespace p138 {

	const int MAXN = 150000;
	int n, p[MAXN], t[MAXN]; // 题目数量,初始分数,做题时间

	void read_case() {
		read(n);
		rep(i, 0, n)
			read(p[i]);
		rep(i, 0, n)
			read(t[i]);
	}

	typedef struct IDA // int data arr
	{
		LL p, t, acc_t, min, max;
	}IDA, * IDA_PTR;
	LL T;
	IDA ida[MAXN];

	bool cmp1(const IDA &i, const IDA &j)
	{
		if (i.t * j.p < i.p * j.t)
			return true;
		else
			return false;
	}

	bool cmp2(const IDA& i, const IDA& j)
	{
		if (i.p < j.p)
			return true;
		else
			return false;
	}

	// 检查mid是否可以使任意最佳做题顺序都不存在悖论
	int check(double mid)
	{
		double max_score = -1e18, _max_score = 1e18;
		for (int i = 0; i < n; i++)
		{
			if (ida[i].p != ida[i - 1].p) _max_score = max_score; // 暂存之前的最高得分
			// 在一组按任何顺序做都不影响最终总分的题目中
			double score = ida[i].p * (1.0 - mid * ida[i].max / T);
			if (score < _max_score) // 如果这道题最后做,发现得分小于之前的最高得分,说明出现悖论
				return false;
			score = ida[i].p * (1.0 - mid * ida[i].min / T); // 越早做得分越高
			max_score = max(max_score, score); // 实时更新最高得分
		}
		return true;
	}

	void solve() {
		for (int i = 0; i < n; i++)
			ida[i].p = p[i];
		for (int i = 0; i < n; i++)
		{
			ida[i].t = t[i];
			T += ida[i].t;
		}

		sort(ida, ida+n, cmp1); // 根据 t[i]/p[i]排序得到的做题顺序是最优的
		for (int i = 0; i < n; i++) // 统计前缀时间,用于计算最终得分
			ida[i].acc_t = ida[i - 1].acc_t + ida[i].t;
		for (int i = 0, j; i < n; i = j)
		{
			for (j = i; j < n && (ida[i].t * ida[j].p == ida[i].p * ida[j].t); j++)
				;
			for (int k = i; k < j; k++) // 对于ti/pi相同的几道题,在这段时间内不论做题顺序如何最终得分之和是一样的
			{
				ida[k].min = ida[i - 1].acc_t + ida[k].t;
				ida[k].max = ida[j - 1].acc_t;
			}
		}
		sort(ida, ida+n, cmp2); // 再根据p排序
		double lo = 0, hi = 1, ans;
		rep(i, 0, 100) {
			double mid = (lo + hi) / 2;
			if (check(mid)) {
				ans = mid;
				lo = mid;
			}
			else
				hi = mid;
		}
		print(ans);
	}
}

namespace p141 {

	const int MAXN = 65;
	int n, a[MAXN];

	void read_case() {
		read(n);
		for (int i = 1; i <= n; i++)
			read(a[i]);
	}

	int done;
	int used[MAXN];
	int total_len, max_len, len;

	void dfs(int tot, int x, int sum)
	{
		if (tot == (total_len / len) - 1) // 剪枝
		{
			done = 1;
			return;
		}
		for (int i = x; i <= n; i++)
		{
			if (!used[i])
			{
				used[i] = 1;
				if (sum + a[i] == len)
				{
					dfs(tot + 1, 1, 0);
					if (done) return;
				}
				else if (sum + a[i] < len)
				{
					dfs(tot, i + 1, sum + a[i]);
					if (done) return;
				}
				used[i] = 0;
				if (sum == 0) // 剪枝
					break;
				while (a[i + 1] == a[i]) // 说明挑选这根木根再加上前面挑选的木棍并继续挑选拼不出一条边，所以跳过后面跟这条木根长度相同的
					i++;
			}
		}
	}

	bool cmp(int x, int y) { return x > y; }

	void solve()
	{
		sort(a + 1, a + n + 1, cmp); // 从大到小排序

		total_len = 0;
		max_len = -1;
		for (int i = 1; i <= n; i++)
		{
			total_len += a[i];
			max_len = max(max_len, a[i]);
		}

		for (len = max_len; len <= total_len; len++) // 
		{
			if (total_len % len != 0)
				continue;

			done = 0;
			memset(used, 0, sizeof(used));

			dfs(0, 1, 0);
			if (done)
			{
				print(len);
				break;
			}
		}
	}	
}

namespace p142 {

	const int MAX_N = 23;

	int n;
	int A[MAX_N], B[MAX_N];

	void read_case() {
		rep(i, 0, n)
			read(A[i], B[i]);
	}

	void init();
	void add_card(int num, int color);
	int find();

	void solve() {
		init();
		rep(i, 0, n)
			add_card(A[i], B[i]);
		print(find());
	}

	typedef struct STATE
	{
		int count[MAX_N];
	}STATE;
	STATE init_state;

	void init()
	{
		memset(&init_state, 0, sizeof(STATE));
	}

	void add_card(int num, int color)
	{
		if (num == 1) // A
			init_state.count[14]++;
		else
			init_state.count[num]++;
	}

	int max_step, ans;
	STATE ans_state;
	vector<STATE> ans_steps;
	void dfs(int cur_step, STATE state, vector<STATE> steps);

	int find()
	{
		max_step = min(n, 13); // 因为最多有13种牌型,所以即便每次只出掉一种牌型,最多13次就可以出光
		ans = 13;
		vector<STATE> steps;
		steps.push_back(init_state);
		dfs(0, init_state, steps);
		//for(int i=0; i<ans_steps.size(); i++)
		//	{
		//	printf("step%d\n",i+1);
		//	print(ans_steps[i]);
		//	}
		return ans;
	}

	int need[4] = {
		0,
		5, // 想出顺子最少需要五种连着的牌型,例如56789.
		3, // 想出连对最少需要三种连着的牌型,例如334455.
		2, // 想出飞机最少需要两种连着的牌型,例如333444.
	};
	int remain(STATE state);

	void dfs(int cur_step, STATE state, vector<STATE> steps)
	{
		if (cur_step >= max_step)
			return;
		for (int i = 3; i >= 1; i--) // 依次检查能否出飞机、连对、顺子
			for (int p = 3; p <= 13; p++)
			{
				int q = p;
				while (q <= 14 && state.count[q] >= i) q++;
				q--; // 找到了从p到q的连着的牌型
				int len = q - p + 1;
				if (len >= need[i]) // 比如找到了3344556677,
				{
					STATE new_state = state;
					for (int k = p; k <= p + need[i] - 2; k++) // 想出连对是必须先出3344
						new_state.count[k] -= i;
					for (int k = p + need[i] - 1; k <= q; k++) // 检查出334455或33445566或33445566771的效果
					{
						new_state.count[k] -= i;
						steps.push_back(new_state);
						dfs(cur_step + 1, new_state, steps);
					}
				}
			}
		int val = cur_step + remain(state);
		if (val < ans)
		{
			ans = val;
			ans_state = state;
			ans_steps = steps;
		}
	}

	int remain(STATE state)
	{
		int b[MAX_N], res = 0;
		memset(b, 0, sizeof(b));
		for (int i = 0; i <= 14; i++) // 统计剩下的牌中牌型的数量,比如AAAA 7788
		{
			if (i == 1)
				continue;
			b[state.count[i]]++; // 统计之后得b[4]=1,b[2]=2
		}

		while (b[4] >= 1 && b[2] >= 2) { b[4]--; b[2] -= 2; res++; }// 四带二对
		while (b[4] >= 1 && b[1] >= 2) { b[4]--; b[1] -= 2; res++; }// 四带二
		while (b[3] >= 1 && b[2] >= 1) { b[3]--; b[2] -= 1; res++; }// 三带对
		while (b[3] >= 1 && b[1] >= 1) { b[3]--; b[1] -= 1; res++; }// 三带一

		res += b[4] + b[3] + b[2] + b[1]; // 出单牌
		return res;
	}	
}

namespace p143 {

	int n, a[5][7]; // 原点在左下角,x轴向右,y轴向上

	void read_case() {
		memset(a, 0, sizeof(a));
		read(n);
		for (int x = 0; x < 5; x++)
		{
			for (int y = 0;; y++)
			{
				int c;
				read(c);
				if (c == 0)
					break;
				a[x][y] = c;
			}
		}
	}

	struct VECTOR3D { int x, y, z; };
	VECTOR3D vec3d(int x, int y, int z) { VECTOR3D v; v.x = x; v.y = y; v.z = z; return v; }
	typedef std::vector<VECTOR3D> STATE;

	int done;
	STATE ans;
	void dfs(int dep, STATE state);

	void solve() {
		STATE s;
		dfs(0, s);

		if (!done)
			printf("-1");
		else
		{
			for (int i = 0; i < ans.size(); i++)
			{
				VECTOR3D& v = ans[i];
				printf("%d %d %d\n", v.x, v.y, v.z);
			}
		}
	}

	int check();
	int fall();
	int erase();

	void dfs(int tot, STATE s)
	{
		if (tot >= n)
		{
			if (check())
			{
				done = 1;
				ans = s;
			}
			return;
		}
		for (int x = 0; x < 5; x++)
			for (int y = 0; y < 7; y++)
			{
				if (a[x][y] == 0)
					continue;
				if (x < 4) // 与右边的交换
				{
					int a_[5][7];
					memcpy(a_, a, sizeof(a));

					swap(a[x][y], a[x + 1][y]);
					fall();
					while (erase())
						fall();

					STATE s1 = s;
					s1.push_back(vec3d(x, y, 1));
					dfs(tot + 1, s1);

					memcpy(a, a_, sizeof(a));
					if (done) return;
				}
				if (x && !a[x - 1][y]) // 当左边为空时与左边交换
				{
					int a_[5][7];
					memcpy(a_, a, sizeof(a));

					swap(a[x][y], a[x - 1][y]);
					fall();
					while (erase())
						fall();

					STATE s1 = s;
					s1.push_back(vec3d(x, y, -1));
					dfs(tot + 1, s1);

					memcpy(a, a_, sizeof(a));
					if (done) return;
				}
			}
	}

	int check()
	{
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 7; j++)
				if (a[i][j])
					return(0);
		return(1);
	}

	int fall() // 如果产生了下落则返回(1)否则返回(0)
	{
		int flag;
		flag = 0;
		for (int x = 0; x < 5; x++)
			for (int y = 1; y < 7; y++)
			{
				if (!a[x][y] || a[x][y - 1])
					continue; // 如果(x,y)是空的或者(x,y)已经落在(x,y-1)上都不需要下落
				flag = 1;
				int y1;
				for (y1 = y - 1; y1 >= 0 && !a[x][y1]; y1--)
					; // y1一直往下直到找到第一个非空的格子
				a[x][y1 + 1] = a[x][y];
				a[x][y] = 0;
			}
		return flag;
	}

	int erase()
	{
		int flag;
		int to_clear[5][7];

		flag = 0;
		for (int x = 0; x < 5; x++)
			for (int y = 0; y < 7; y++)
				to_clear[x][y] = 0;

		for (int x = 0; x < 3; x++)
			for (int y = 0; y < 7; y++)
			{
				if (!a[x][y])
					continue;
				int x1;
				for (x1 = x; x1 < 5 && a[x1][y] == a[x][y]; x1++)
					; // x1一直往右直到找到第一个跟a[x][y]不同的格子
				if (x1 - x >= 3) // 连续三个以上可以消除
				{
					flag = 1;
					for (int x1 = x; x1 < 5 && a[x1][y] == a[x][y]; x1++)
						to_clear[x1][y] = 1;

				}
			}
		for (int x = 0; x < 5; x++)
			for (int y = 0; y < 5; y++)
			{
				if (!a[x][y])
					continue;
				int y1;
				for (y1 = y; y1 < 7 && a[x][y1] == a[x][y]; y1++)
					; // y1一直往上直到找到第一个跟a[x][y]不同的格子
				if (y1 - y >= 3)
				{
					flag = 1;
					for (y1 = y; y1 < 7 && a[x][y1] == a[x][y]; y1++)
						to_clear[x][y1] = 1;
				}
			}

		for (int x = 0; x < 5; x++)
			for (int y = 0; y < 7; y++)
			{
				if (to_clear[x][y])
					a[x][y] = 0;
			}

		return flag;
	}
}

namespace p144 {
	
	const int MAXW = 20;
	const int MAXH = 20;

	const int BLOCK = 1;
	const int S = 2;
	const int G = 3;

	int w, h;
	int a[MAXW][MAXH];
	int sx, sy;

	void read_case() {
		read(w, h);
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++)
			{
				read(a[x][y]);
				if (a[x][y] == S)
				{
					sx = x; sy = y;
				}
			}
	}

	struct VECTOR3D { int x, y, z; };
	VECTOR3D vec3d(int x, int y, int z) { VECTOR3D v; v.x = x; v.y = y; v.z = z; return v; }
	typedef std::vector<VECTOR3D> STATE;

	const int INF = 1000000000;
	int ans;
	void dfs(int tot, int x, int y, STATE s);

	void solve()
	{
		ans = INF;

		STATE s;
		dfs(0, sx, sy, s);

		if (ans != INF)
			printf("%d\n", ans);
		else
			printf("-1\n");
	}

	int dir[4][2] =
	{
	{0,-1},
	{1,0},
	{0,1},
	{-1,0}
	};

	void dfs(int tot, int x, int y, STATE s)
	{
		if (tot > min(10, ans))
			return;

		for (int d = 0; d < 4; d++)
		{
			int nx = x + dir[d][0], ny = y + dir[d][1];
			if (nx<0 || nx>w - 1 || ny<0 || ny>h - 1)
				continue;
			if (a[nx][ny] == BLOCK)
				continue;
			int cx, cy;
			for (;;)
			{
				cx = nx; cy = ny;
				if (a[cx][cy] == G)
				{
					if (tot + 1 < ans && tot + 1 <= 10)
						ans = tot + 1;
					break;
				}
				nx = cx + dir[d][0]; ny = cy + dir[d][1];
				if (nx<0 || nx>w - 1 || ny<0 || ny>h - 1)
				{
					break;
				}
				if (a[nx][ny] == BLOCK)
				{
					a[nx][ny] = 0;
					STATE s1 = s;
					//s1.push_back(vec3d(x,y,d));
					dfs(tot + 1, cx, cy, s1);
					a[nx][ny] = BLOCK;
					break;
				}
			}
		}
	}
}

namespace p145 {
	
	const int MAXM = 29;
	int N, m;

	void read_case() {
		scanf("%d%d", &N, &m);
	}

	struct VECTOR2D { int x, y; };
	VECTOR2D vec2d(int x, int y) { VECTOR2D v; v.x = x; v.y = y; return v; }
	typedef std::vector<VECTOR2D> STATE;

	const int INF = 1000000000;
	int ans;
	int low_s[MAXM], low_v[MAXM];
	void dfs(int dep, int S, int V, int max_r, int max_h, STATE state);

	void solve()
	{
		// 计算第1开始到第m层的最小表面积和体积
		for (int i = 1; i <= m; i++)
		{
			low_s[i] = low_s[i - 1] + 2 * i * i;
			low_v[i] = low_v[i - 1] + i * i * i;
		}

		ans = INF;

		STATE s;
		dfs(m, 0, 0, 100, 10000, s);
		if (ans == INF)
			printf("0\n");
		else
			printf("%d\n", ans);
	}

	void dfs(int dep, int S, int V, int max_r, int max_h, STATE s)
	{
		if (dep == 0)
		{
			if (V == N && S < ans)
			{
				ans = S;
			}
			return;
		}
		if (V > N || S > ans) // 剪枝
			return;

		if ((V + low_v[dep]) > N || (S + low_s[dep] > ans)) // 如果剩下的层都用最小体积的蛋糕...
			return;

		if ((2 * (N - V) / max_r + S) > ans) // 重要!如果把剩下的体积做成一个蛋糕所得的表面积
			return;

		for (int r = max_r - 1; r >= dep; r--)
		{
			if (dep == m) S = r * r;
			for (int h = max_h - 1; h >= dep; h--)
			{
				//STATE s1 = s;
				//s1.push_back(vec2d(r,h));
				dfs(dep - 1, S + 2 * r * h, V + r * r * h, r, h, s);
			}
		}
	}
}

namespace p146 {

	const int MAXX = 6;
	const int MAXY = 6;
	int a[6][6];

	void read_case() {
		for (int y = 0; y < 6; y++)
			for (int x = 0; x <= y; x++)
				read(a[x][y]);
	}

	void init();
	int bfs();

	void solve() {
		init();
		int ans = bfs();
		if (ans > 20)
			printf("too difficult\n");
		else
			printf("%d\n", ans);
	}

	typedef long long ll;
	typedef struct STATE
	{
		int tot;
		int a[6][6], x, y;
	}STATE, * STATE_PTR;

	STATE st, ed;
	std::queue<STATE> q;
	std::unordered_map<ll, int> M, N;
	int ans;
	STATE final_state;

	void init()
	{
		while (!q.empty())
			q.pop();
		M.clear();
		N.clear();
		ans = 21;
	}

	int dir[4][2] =
	{
	{-1,-1},
	{0,-1},
	{0,1},
	{1,1}
	};
	ll hash(STATE_PTR state);

	int bfs()
	{
		for (int y = 0; y < 6; y++)
			for (int x = 0; x <= y; x++)
				st.a[x][y] = y;
		st.x = st.y = 0;

		for (int y = 0; y < 6; y++)
			for (int x = 0; x <= y; x++)
			{
				ed.a[x][y] = a[x][y];
				if (ed.a[x][y] == 0)
				{
					ed.x = x;
					ed.y = y;
				}
			}

		q.push(st);
		M[hash(&st)] = 0;
		while (!q.empty())
		{
			STATE s = q.front(); q.pop();
			for (int d = 0; d < 4; d++)
			{
				int nx = s.x + dir[d][0], ny = s.y + dir[d][1];
				if (ny >= 0 && ny < 6 && nx >= 0 && nx <= ny) // 第y行的x坐标范围是[0,y]
				{
					STATE s1 = s;
					swap(s1.a[s1.x][s1.y], s1.a[nx][ny]);
					s1.x = nx; s1.y = ny;
					s1.tot++;
					ll ha = hash(&s1);
					if (M.find(ha) == M.end() && s1.tot <= 10)
					{
						M[ha] = s1.tot;
						q.push(s1);
					}
				}
			}
		}

		ll ha = hash(&ed);
		N[ha] = 0;
		q.push(ed);
		if (M.find(ha) != M.end())
		{
			ans = M[ha];
			//final_state = ed;
			return ans;
		}

		while (!q.empty())
		{
			STATE s = q.front(); q.pop();
			for (int d = 0; d < 4; d++)
			{
				int nx = s.x + dir[d][0], ny = s.y + dir[d][1];
				if (ny >= 0 && ny < 6 && nx >= 0 && nx <= ny) // 第y行的x坐标范围是[0,y]
				{
					STATE s1 = s;
					swap(s1.a[s1.x][s1.y], s1.a[nx][ny]);
					s1.x = nx; s1.y = ny;
					s1.tot++;
					ll ha = hash(&s1);
					if (N.find(ha) == N.end() && s1.tot <= 10)
					{
						N[ha] = s1.tot;
						q.push(s1);
						if (M.find(ha) != M.end())
						{
							ans = min(ans, s1.tot + M[ha]);
							//final_state = s1;
						}
					}
				}
			}
		}

		return ans;
	}

	ll hash(STATE_PTR state)
	{
		ll val = 0;
		for (int y = 0; y < 6; y++)
			for (int x = 0; x <= y; x++)
				val = 6LL * val + state->a[x][y];
		return val;
	}
}

namespace p151 {

	const int MAXN = 300000;
	int n, m, q, x[MAXN + 1], y[MAXN + 1];

	void read_case() {
		read(n, m, q);
		rep(i, 1, q + 1)
			read(x[i], y[i]);
	}

	typedef long long ll;

	class Interval // 鍔ㄦ�佸紑鐐?
	{
	public:
		int l, r, sum;
		Interval* lc, * rc;
		Interval() {}
		Interval(int l, int r)
		{
			this->l = l; this->r = r;
			lc = rc = NULL;
			sum = 0;
		}
		void update()
		{
			sum = 0;
			if (lc != NULL)
				sum += lc->sum;
			if (rc != NULL)
				sum += rc->sum;
		}
		void change(int p, int v)
		{
			if (l == r)
			{
				sum = v;
				return;
			}
			int mid = (l + r) / 2;
			if (lc == NULL)
				lc = new Interval(l, mid);
			if (p <= lc->r)
				lc->change(p, v);
			else
			{
				if (rc == NULL)
					rc = new Interval(mid + 1, r);
				rc->change(p, v);
			}

			update();
		}
		int query(int k)
		{
			if (l == r)return l;
			int mid = (l + r) / 2;
			if (lc == NULL)
				lc = new Interval(l, mid);

			int tot = mid - l + 1 - lc->sum;
			if (k <= tot)
				return lc->query(k);
			else
			{
				if (rc == NULL)
					rc = new Interval(mid + 1, r);
				return rc->query(k - tot);
			}
		}
	};

	Interval* row[MAXN + 1], * col;
	vector<ll> a[MAXN + 1], b;

	void solve() {
		rep(i, 1, n + 1)
		{
			row[i] = new Interval(1, m + q);
		}
		col = new Interval(1, n + q);
		rep(i, 1, q + 1)
		{
			if (y[i] == m)
			{
				int j = col->query(x[i]);
				ll id;
				if (j <= n)
					id = 1LL * j * m;
				else
					id = b[j - n - 1];
				col->change(j, 1);
				col->change(n + i + 1, 0);
				b.push_back(id);
				print(id);
				print();
			}
			else
			{
				int j = row[x[i]]->query(y[i]);
				ll id1, id2;
				if (j <= m - 1)
					id1 = 1LL * (x[i] - 1) * m + j;
				else
					id1 = a[x[i]][j - (m - 1) - 1];
				row[x[i]]->change(j, 1);
				row[x[i]]->change((m - 1) + i + 1, 0);
				j = col->query(x[i]);
				if (j <= n)
					id2 = 1LL * j * m;
				else
					id2 = b[j - n - 1];
				col->change(j, 1);
				col->change(n + i + 1, 0);
				a[x[i]].push_back(id2);
				b.push_back(id1);
				print(id1);
				print();
			}
		}
	}
}

namespace p152_part1 {

	const int MAXN = 100009;
	int n, a[MAXN], b[MAXN];

	void read_case() {
		scanf("%d", &n);
		for (int i = 1; i <= n; i++)
			scanf("%d", &a[i]);
		for (int i = 1; i <= n; i++)
			scanf("%d", &b[i]);
	}

	typedef struct VECTOR3
	{
		int x, y, z;
		bool operator<(const VECTOR3& other) const
		{
			return other.z < z;
		}
	}VECTOR3;
	priority_queue<VECTOR3> q;

	void solve() {
		for (int i = 1; i <= n; i++)
		{
			VECTOR3 vec;
			vec.x = i;
			vec.y = 1;
			vec.z = a[i] + b[1];
			q.push(vec);
		}

		for (int k = 1; k <= n; k++)
		{
			VECTOR3 vec = q.top(); q.pop();
			int x, y, z;
			printf("%d ", vec.z);
			vec.y++;
			vec.z = a[vec.x] + b[vec.y];
			q.push(vec);
		}
	}
}

namespace p152_part2 {

	void read_case() {

	}

	typedef long long ll;
	void init();
	void next(ll* p);

	void solve() {
		ll ans;

		init();
		for (int i = 1; i <= 1500; i++)
			next(&ans);
		printf("The 1500'th ugly number is %lld.\n", ans);
	}

	struct cmp
	{
		bool operator()(ll i, ll j)
		{
			return i > j;
		}
	};
	std::priority_queue<ll, std::vector<ll>, cmp> q;

	void init()
	{
		q.push(1LL);
	}

	void next(ll* num)
	{
		*num = q.top();
		q.push(*num * 2LL);
		q.push(*num * 3LL);
		q.push(*num * 5LL);
		while (q.top() == *num)
			q.pop();
	}
}

namespace p162 {

	const int MAXN = 100001;
	int n, h[MAXN];

	void read_case() {
		scanf("%d", &n);
		for (int i = 1; i <= n; i++)
			scanf("%d", &h[i]);
	}

	struct
	{
		//STATE state;
		int len;
	} f[MAXN][2]; // f[i][0]记录点i为波谷的最长序列 f[i][1]记录点i为波峰的最长序列

	int ans;

	void solve()
	{
		/*f[1][0].state.push_back(h[1]);*/ f[1][0].len = 1;
		/*f[1][1].state.push_back(h[1]);*/ f[1][1].len = 1;
		for (int i = 2; i <= n; i++)
		{
			for (int j = max(1, i - 1000); j < i; j++)
			{
				if (h[j] < h[i] && f[j][1].len + 1 > f[i][0].len) // 形成波谷
				{
					f[i][0] = f[j][1];
					//f[i][0].state.push_back(h[i]);
					f[i][0].len++;
				}
				if (h[j] > h[i] && f[j][0].len + 1 > f[i][1].len)
				{
					f[i][1] = f[j][0];
					//f[i][1].state.push_back(h[i]);
					f[i][1].len++;
				}
			}
		}



		int ans = -1;
		for (int i = 1; i <= n; i++)
			ans = max(ans, max(f[i][0].len, f[i][1].len));

		printf("%d\n", ans);
	}
}

namespace p163 {

	const int  MAXN = 2509;
	const int MAX_Q = 100000;
	const int  PLACE =1;
	const int  REMOVE =2;
	const int  WALK =3;

	int n, m, q;
	int op[MAX_Q], r1[MAX_Q], c1[MAX_Q], r2[MAX_Q], c2[MAX_Q];

	void read_case() {
		scanf("%d%d%d", &n, &m, &q);
		rep(i,0,q)
			scanf("%d%d%d%d%d", &op[i], &r1[i], &c1[i], &r2[i], &c2[i]);
	}

	void add_fence(int r1, int c1, int r2, int c2);
	void remove_fence(int r1, int c1, int r2, int c2);
	int walk(int r1, int c1, int r2, int c2);

	void solve() {
		rep(i, 0, q) {
			switch (op[i]) {
			case PLACE:
				add_fence(r1[i], c1[i], r2[i], c2[i]);
				break;
			case REMOVE:
				remove_fence(r1[i], c1[i], r2[i], c2[i]);
				break;
			case WALK:
				if (walk(r1[i], c1[i], r2[i], c2[i]))
					printf("Yes\n");
				else
					printf("No\n");
				break;
			default:
				break;
			}
		}
		
	}

	typedef long long ll;
	typedef struct FENCE
	{
		int r1, c1, r2, c2;
	}FENCE, * FENCE_PTR;
	unordered_map<ll, int> M;

	mt19937 random;
	ll hash(FENCE_PTR fence);
	void add(int r, int c, int v);
	int sum(int r, int c);

	void add_fence(int r1, int c1, int r2, int c2)
	{
		int v = random();

		FENCE fence;
		fence.r1 = r1;
		fence.c1 = c1;
		fence.r2 = r2;
		fence.c2 = c2;
		M[hash(&fence)] = v;

		add(r1, c1, v);
		add(r1, c2 + 1, -v);
		add(r2 + 1, c1, -v);
		add(r2 + 1, c2 + 1, v);
	}

	void remove_fence(int r1, int c1, int r2, int c2)
	{
		FENCE fence;
		fence.r1 = r1;
		fence.c1 = c1;
		fence.r2 = r2;
		fence.c2 = c2;
		int v = M[hash(&fence)];
		add(r1, c1, -v);
		add(r1, c2 + 1, v);
		add(r2 + 1, c1, v);
		add(r2 + 1, c2 + 1, -v); // 注意!当r2=N,c2=M时,再加1就会越界,所以数组要开大点
	}

	int walk(int r1, int c1, int r2, int c2)
	{
		int v1 = sum(r1, c1);
		int v2 = sum(r2, c2);
		return v1 == v2;
	}

	int bit[MAXN][MAXN];
	void add(int r, int c, int v)
	{
		for (int i = r; i <= n; i += i & -i)
			for (int j = c; j <= m; j += j & -j)
				bit[i][j] += v;
	}

	int sum(int r, int c)
	{
		int res = 0;
		for (int i = r; i >= 1; i -= i & -i)
			for (int j = c; j >= 1; j -= j & -j)
				res += bit[i][j];
		return res;
	}

	ll hash(FENCE_PTR fence)
	{
		return fence->r1 * 15643757501LL + fence->c1 * 6255001LL + fence->r2 * 2501 + fence->c2;
	}
}

namespace p171 {

	struct SqrtDecomp {
		struct Block {
			int l, r;
			VI v;
		};
		vector<Block> blocks;
		VI a;
		int n;
		int sz;//块大小

		SqrtDecomp(VI &a_, int n_) {
			a = a_;
			n = n_;
			sz = (int)ceil(sqrt(n));
			for(int l=1; l<=n; l+=sz) {
				int r = min(l + sz - 1, n);
				Block block;
				block.l = l;
				block.r = r;
				rep(i, l, r+1)
					block.v.push_back(i);
				blocks.push_back(block);
			}
		}

		// 计算交换后会产生的逆序对的数量
		int calc(int l, int r) {
			int res = 0;

			LL cnt1 = query(l + 1, r - 1, a[l]);
			LL cnt2 = query(l + 1, r - 1, a[r]);

			//设
			//x1表示区间内小于a[l]的数的个数
			//y1表示区间内大于a[l]的数的个数
			//x2表示区间内小于a[r]的数的个数
			//y2表示区间内大于a[r]的数的个数
			//有
			//x1 + y1 = r-l+1
			//x2 + y2 = r-l+1
			//交换后
			//ans = (y1-x1) + (x2-y2) = ... = 2*(y1-y2)

			res = 2 * (cnt1 - cnt2);
			if (a[l] > a[r])
				res--;
			else // a[l]<a[r]
				res++;

			return res;
		}

		// find the number of integers which greater than s
		LL query(int l, int r, int v)
		{
			int b1 = (l-1) / sz;
			int b2 = (r-1) / sz;

			LL cnt = 0;
			if (b1 == b2) {
				rep(i, l, r + 1)
					if (a[i] > v)
						cnt++;
			}
			else {
				rep(i, l, blocks[b1].r + 1) {
					if (a[i] > v)
						cnt++;
				}
				rep(i, b1+1, b2) {
					cnt += blocks[i].v.end() - upper_bound(all(blocks[i].v), v);
				}
				rep(i, blocks[b2].l, r + 1) {
					if (a[i] > v)
						cnt++;
				}
			}

			return cnt;
		}

		void change(int l, int r) {
			int b1 = (l - 1) / sz;
			int b2 = (r - 1) / sz;
			blocks[b1].v.erase(lower_bound(all(blocks[b1].v), a[l]));
			blocks[b1].v.insert(upper_bound(all(blocks[b1].v), a[r]), a[r]);
			blocks[b2].v.erase(lower_bound(all(blocks[b2].v), a[r]));
			blocks[b2].v.insert(upper_bound(all(blocks[b2].v), a[l]), a[l]);
			swap(a[l], a[r]);
		}
	};

	const int MAXN = 200000;
	const int MAX_Q = 50000;
	int n, q, L[MAX_Q], R[MAX_Q];

	void read_case() {
		read(n, q);
		rep(i, 0, q)
			read(L[i], R[i]);
	}

	void solve() {
		VI a = VI(n+1);
		rep(i, 1, n + 1)
			a[i] = i;
		SqrtDecomp decomp(a, n);
		LL ans = 0;
		rep(i, 0, q)
		{
			int l = L[i];
			int r = R[i];
			if (l == r)
			{
				print(ans);
				continue;
			}
			if (l > r) // NOTE!!!
				swap(l, r);

			ans += decomp.calc(l, r);
			decomp.change(l, r); // 交换
			print(ans);
		}
	}
}

namespace p172 {

	const int MAX_N = 100000, MAX_M = 100000;
	int n, c, m; // 读取文章字数、汉字种类数(即最大编号)、询问数
	int a[MAX_N], L[MAX_M], R[MAX_M];

	void read_case() {
		read(n, c, m);
		rep(i, 0, n)
			read(a[i]);
		rep(i, 0, m)
			read(L[i], R[i]);
	}

	// 不知道为什么用map<int,int>会TLE

	struct SqrtDecomp { // 因为该struct只是对平方分割算法的简单封装,所以直接使用全局变量
		struct Block {
			int l, r;
		};
		vector<Block> blocks;
		int sz;//块大小

		vector<VI> h; // h[i]保存字符i出现的位置
		vector<VI> f; // f[i][j]表示第i块到第j块的答案

		SqrtDecomp() {
			sz = (int)ceil(sqrt(n));
			for (int l = 1; l <= n; l += sz) {
				int r = min(l + sz - 1, n);
				Block block = { l,r };
				blocks.push_back(block);
			}

			h = vector<VI>(c + 1);
			rep(i, 1, n)
				h[a[i]].push_back(i);

			int m = blocks.size(); // 块总数
			f = vector<VI>(m, VI(m));

			rep(i, 0, m) {
				VI cnt(c + 1);
				int res = 0;
				rep(j, i, m) {
					rep(k, blocks[j].l, blocks[j].r + 1) { //注意这里是j
						if (cnt[a[k]] > 0 && cnt[a[k]] % 2 == 0)
							res -= 1; // 偶变奇
						else if (cnt[a[k]] % 2 == 1)
							res += 1; // 奇变偶
						cnt[a[k]]++;
					}
					f[i][j] = res;
				}
			}
		}

		int query(int l, int r)
		{
			int b1 = (l - 1) / sz;
			int b2 = (r - 1) / sz;

			int res = 0;
			if (b1 == b2 || b1 + 1 == b2) { // 如果在同一块或在相邻块
				VI mp(c + 1);
				rep(i, l, r + 1)
					mp[a[i]]++;
				rep(i, 0, c + 1)
					if (mp[i] > 0 && mp[i] % 2 == 0)
						res++;
			}
			else {
				VI mp(c + 1);
				VI key;
				rep(i, l, blocks[b1].r + 1) {
					if (mp[a[i]]++ == 0)
						key.push_back(a[i]);
				}
				rep(i, blocks[b2].l, r + 1) {
					if (mp[a[i]]++ == 0)
						key.push_back(a[i]);
				}
				res = f[b1 + 1][b2 - 1];
				rep(i, 0, key.size()) {
					int c = key[i];
					int cnt = upper_bound(all(h[c]), blocks[b2].l - 1) - // 找出字符c在块b1到b2之间的出现次数
						lower_bound(all(h[c]), blocks[b1].r + 1);
					if (cnt == 0) { // 如果c没出现
						if (mp[c] % 2 == 0)
							res++;
					}
					else {
						if (cnt % 2 == 0) { // 如果本来出现了偶数次,即c已被算入res
							if (mp[c] % 2 == 0) { // 如果在两端也出现了偶数次,那么c仍然出现了偶数次

							}
							else { // 如果在两端出现了奇数次,那么c实际出现了奇数次
								res--;
							}
						}
						else if (cnt % 2 == 1) { // 如果本来出现了奇数次,即c未被算入res
							if (mp[c] % 2 == 0) { // 如果在两端出现了偶数次

							}
							else { // 如果在两端出现了奇数次
								res++;
							}
						}
					}
				}
			}

			return res;
		}
	};

	void solve() {
		VI t;
		t.push_back(0);
		rep(i, 0, n)
			t.push_back(a[i]);

		SqrtDecomp sd; // 平方分割

		int ans = 0;
		rep(i,0,m) {
			int l, r, tl, tr; 
			l = L[i];
			r = R[i];
			tl = (l + ans) % n + 1;
			tr = (r + ans) % n + 1;
			if (tl > tr)
				swap(tl, tr);
			ans = sd.query(tl, tr);
			print(ans);
		}
	}
}

namespace p223_part2 {

	const int MAXN = 101;

	int n, m, a[MAXN][MAXN];

	void read_case() {
		read(n, m);
		for (int i = 1; i <= n; i++)
			for (int j = 1; j <= m; j++)
				read(a[i][j]);
	}

	const int INF = 1000000000;
	int f[MAXN][MAXN];
	void dp(int i, int j);

	void solve()
	{
		for (int i = 1; i <= n; i++)
			for (int j = 1; j <= m; j++)
				f[i][j] = INF;

		int ans = -1;
		for (int i = 1; i <= n; i++)
			for (int j = 1; j <= m; j++)
			{
				dp(i, j);
				if (f[i][j] > ans)
					ans = f[i][j];
			}
		print(ans);
	}

	int dir[4][2] =
	{
	{-1,0},
	{0,1},
	{1,0},
	{0,-1}
	};

	void dp(int r, int c)
	{
		if (f[r][c] != INF)
			return;

		f[r][c] = 1;
		for (int d = 0; d < 4; d++)
		{
			int nr = r + dir[d][0];
			int nc = c + dir[d][1];
			if (nr >= 1 && nr <= n && nc >= 1 && nc <= m && a[nr][nc] < a[r][c])
			{
				dp(nr, nc);
				if (f[nr][nc] + 1 > f[r][c])
					f[r][c] = f[nr][nc] + 1;
			}
		}
	}
}

namespace p231 {

	const int MAXN = 1000;

	int c, n;
	int m[MAXN][MAXN];

	void read_case() {
		scanf("%d", &c);
		scanf("%d", &n);
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++)
				scanf("%d", &m[i][j]);
			for (int j = n - 1; j > 0; j--)
				m[i][j] = m[i][j] - m[i][j - 1];
		}
	}

	int mem[MAXN][MAXN];

	int dp(int i, int j, int cost)
	{
		if (i == n)
			return cost;
		int res = 0x3f3f3f3f;
		if (mem[i][j] != -1)
			res = cost + mem[i][j];
		else
		{
			res = min(res, dp(i + 1, j, cost + m[j][i]));
			res = min(res, dp(i + 1, i, cost + c + m[i][i]));
			mem[i][j] = res - cost;
		}
		return res;
	}

	void solve() {
		for (int i = 0; i < MAXN; i++)
			for (int j = 0; j < MAXN; j++)
				mem[i][j] = -1;
		printf("%d\n", dp(1, 0, c + m[0][0]));
	}
}

namespace p232_part1 {

	const int MAXN = 101;
	int n, a[MAXN];

	void read_case() {
		read(n);
		for (int i = 1; i <= n; i++) // 使用1~n，避免处理坐标小于0的情况
			read(a[i]);
	}

	const int INF = 1000000000;
	int f[MAXN][MAXN], g[MAXN][MAXN]; // 从第i堆起合并j堆的最小/最大花费
	int cost[MAXN][MAXN]; // 从第i堆起合并j堆的花费

	void solve()
	{
		for (int i = 1; i <= n; i++) // 初始条件
			cost[i][1] = a[i];
		for (int j = 2; j <= n; j++)
			for (int i = 1; i <= n; i++) {
				int s = i + 1;// 另一区间的起点
				if (s > n) s -= n;
				cost[i][j] = a[i] + cost[s][j - 1];
			}

		for (int j = 2; j <= n; j++)
		{
			for (int i = 1; i <= n; i++)
			{
				f[i][j] = INF;
				g[i][j] = -INF;
				for (int k = 1; k <= j - 1; k++)
				{
					int s = i + k; // 另一区间的起点
					if (s > n) s -= n;
					f[i][j] = min(f[i][j], f[i][k] + f[s][j - k] + cost[i][j]);
					g[i][j] = max(g[i][j], g[i][k] + g[s][j - k] + cost[i][j]);
				}
			}
		}

		int minc = INF, maxc = -INF;
		for (int i = 1; i <= n; i++)
		{
			minc = min(f[i][n], minc);
			maxc = max(g[i][n], maxc);
		}
		printf("%d\n%d\n", minc, maxc);

	}
}

namespace p233_part1 {
	
	const int MAXN = 5001;
	int n;
	char s[MAXN];

	void read_case() {
		scanf("%d", &n);
		scanf("%s", s + 1);
	}

	//要用short，不然会MLE
	const short INF = 0x3f3f;
	int f[MAXN][MAXN]; // f[i][j]表示使s[i~j]变成回文串所需添加的字符
	void dp(int, int);

	void solve()
	{
		for (int i = 0; i <= n; i++)
			for (int j = 0; j <= n; j++)
				f[i][j] = INF;
		dp(1, n);
		printf("%d\n", f[1][n]);
	}

	void dp(int i, int j) // 使[i,j]成为回文串最少需要插入的字符数量
	{
		if (i >= j) // 可能由i+1,j-1转移来时i+1=j,所以此时i>j了
		{
			f[i][j] = 0;
			return;
		}

		if (f[i][j] != INF)
			return;

		if (s[i] == s[j])
		{
			dp(i + 1, j - 1);
			f[i][j] = f[i + 1][j - 1];
		}
		else
		{
			dp(i, j - 1);
			dp(i + 1, j);
			if (f[i][j - 1] < f[i + 1][j])
			{
				f[i][j] = f[i][j - 1];
				f[i][j]++;
			}
			else
			{
				f[i][j] = f[i + 1][j];
				f[i][j]++;
			}
		}
	}
}

namespace p234 {

	const int MAXN = 101;

	int n;
	char sa[MAXN], sb[MAXN];

	void read_case() {
		scanf("%s", sa + 1);
		scanf("%s", sb + 1);
	}

	const int INF = 1000000000;
	int f[MAXN], g[MAXN][MAXN]; // f[i]记录将sa[1~i]染成sb[1~i]的操作 g[i][j]记录将空串染成sb[i~j]的操作
	void dp(int i, int j);

	void solve()
	{
		n = strlen(sa + 1);

		for (int i = 1; i <= n; i++)
			for (int j = 1; j <= n; j++)
				g[i][j] = INF;

		for (int i = 1; i <= n; i++)
			f[i] = INF;

		dp(1, n);

		for (int i = 1; i <= n; i++)
		{
			if (sa[i] == sb[i])
			{
				f[i] = f[i - 1];
				continue;
			}
			for (int j = 0; j < i; j++)
			{
				if (f[j] + g[j + 1][i] < f[i])
				{
					f[i] = f[j];
					f[i] += g[j + 1][i];
				}
			}
		}

		printf("%d\n", f[n]);
	}

	void dp(int i, int j)
	{
		if (i == j)
		{
			g[i][j] = 1;
			return;
		}

		if (g[i][j] != INF)
			return;

		if (sb[i] == sb[i + 1] || sb[i] == sb[j])
		{
			dp(i + 1, j);
			g[i][j] = g[i + 1][j];
		}
		else
		{
			dp(i + 1, j);
			g[i][j] = g[i + 1][j];
			g[i][j]++;
		}

		if (sb[j - 1] == sb[j] || sb[i] == sb[j])
		{
			dp(i, j - 1);
			if (g[i][j - 1] < g[i][j])
			{
				g[i][j] = g[i][j - 1];
			}
		}
		else
		{
			dp(i, j - 1);
			if (g[i][j - 1] + 1 < g[i][j])
			{
				g[i][j] = g[i][j - 1];
				g[i][j]++;
			}
		}

		for (int k = i; k < j; k++)
		{
			dp(i, k);
			dp(k + 1, j);
			if (g[i][k] + g[k + 1][j] < g[i][j])
			{
				g[i][j] = g[i][k];
				g[i][j] += g[k + 1][j];
			}
		}
	}
}

namespace p235 {
	
	const int MAXN = 2009;
	const int MAXK = 1009;
	int n, k, a[MAXN];

	void read_case() {
		read(n, k);
		for (int i = 1; i <= n; i++)
			read(a[i]);
	}

	const int INF = 1000000000;
	int f[MAXN][MAXK]; // f[i][j]记录从前i件物品中选k对的最小疲劳度

	void solve()
	{
		sort(a + 1, a + n + 1);

		for (int i = 1; i <= n; i++) // 再设置为无穷大,不然因f[i-1][j]为0总会取到它
			for (int j = 1; j <= k; j++)
			{
				f[i][j] = INF;
			}

		for (int i = 2; i <= n; i++)
			for (int j = 1; j * 2 <= i; j++)
			{
				if (f[i - 1][j] < f[i - 2][j - 1] + (a[i] - a[i - 1]) * (a[i] - a[i - 1]))
				{
					f[i][j] = f[i - 1][j];
				}
				else
				{
					f[i][j] = f[i - 2][j - 1];
					f[i][j] += (a[i] - a[i - 1]) * (a[i] - a[i - 1]);
				}
			}

		printf("%d\n", f[n][k]);
	}

}

namespace p236 {

	int n, m;
	int a[10000];
	
	void read_case() {
		read(n, m);
		for (int i = 0; i < n; i++)
			read(a[i]);
	}

	int g[10000][101], l[10000][101], mem[10000][101];

	int dfs(int i, int j, int sum)
	{
		int res = 0x3f3f3f3f;
		if (i == n)
			res = sum;
		else if (mem[i][j] != -1)
			res = sum + mem[i][j];
		else if (a[i] != -1)
			res = dfs(i + 1, j, sum);
		else
		{
			for (int k = j; k < m + 1; k++)
				res = min(res, dfs(i + 1, k, sum + g[i][k] + l[i][k]));
			mem[i][j] = res - sum;
		}
		return res;
	}

	void solve() {
		for (int i = 1; i < n; i++)
		{
			for (int j = 1; j <= m; j++)
			{
				g[i][j] = g[i - 1][j];
				if (a[i - 1] != -1 && a[i - 1] > j)
					g[i][j]++;
			}
		}

		for (int i = n - 2; i > -1; i--)
		{
			for (int j = 1; j <= m; j++)
			{
				l[i][j] = l[i + 1][j];
				if (a[i + 1] != -1 && a[i + 1] < j)
					l[i][j]++;
			}
		}

		int t = 0;
		for (int i = 1; i < n; i++)
			if (a[i] != -1)
				t += g[i][a[i]];

		for (int i = 0; i < 10000; i++)
			for (int j = 0; j < 101; j++)
				mem[i][j] = -1;

		int ans = dfs(0, 1, 0);
		print(t + ans);
	}
}

namespace p241_part1 {

	const int MAX_N = 1000, MAX_W = 1000;

	int n, W;
	int w[MAX_N], v[MAX_N];
	int dp[MAX_N + 1][MAX_W + 1];

	void read_case() {
		read(n, W);
		rep(i, 1, n+1)
			read(w[i], v[i]);
	}

	void solve() {
		rep(i, 1, n+1)
			rep(j, 0, W + 1)
				if (j < w[i])
					dp[i][j] = dp[i - 1][j];
				else
					dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i]] + v[i]);
		cout << dp[n][W];
	}
}

namespace p241_part2 {

	const int MAX_N = 1000, MAX_W = 1000;

	int n, W;
	int w[MAX_N], v[MAX_N];
	int dp[MAX_N + 1][MAX_W + 1];

	void read_case() {
		read(n, W);
		rep(i, 1, n + 1)
			read(w[i], v[i]);
	}

	void solve() {
		rep(i, 1, n+1)
			rep(j, 0, W + 1)
			if (j < w[i])
				dp[i][j] = dp[i-1][j];
			else
				dp[i][j] = max(dp[i-1][j], dp[i][j - w[i]] + v[i]);
		cout << dp[n][W];
	}
}

namespace p242_part1 {

	const int MAX_N = 100, MAX_W = 100;

	int n, W;
	int w[MAX_N], v[MAX_N], c[MAX_N];
	int dp[MAX_N + 1][MAX_W + 1];

	void read_case() {
		read(n, W);
		rep(i, 1, n + 1)
			read(w[i], v[i], c[i]);
	}

	void solve() {
		rep(i, 1, n + 1)
			rep(j, 0, W + 1)
				rep(k, 0, min(j / w[i], c[i])+1) // 这里注意是W[i]而不是W[i+1],因为数组下标从0开始
					dp[i][j] = max(dp[i][j], dp[i - 1][j - k * w[i]] + k * v[i]);

		cout << dp[n][W];
	}
}

namespace p242_part2 {

	const int MAX_N = 1000, MAX_W = 2000;

	int n, W;
	int w[MAX_N], v[MAX_N], c[MAX_N];
	int dp[MAX_N + 1][MAX_W + 1];

	void read_case() {
		read(n, W);
		rep(i, 1, n + 1)
			read(w[i], v[i], c[i]);
	}

	void solve() {
		int j = n;
		rep(i, 1, n + 1) {
			rep(k, 0, 31) {
				int t = 1 << k;
				if (t < c[i]) {
					j++;
					w[j] = t * w[i];
					v[j] = t * v[i];
					c[i] -= t;
				}
				else {
					j++;
					w[j] = c[i] * w[i];
					v[j] = c[i] * v[i];
					c[i] = 0;
					break;
				}
			}
		}
		n = j;
		rep(i, 1, n + 1)
			rep(j, 0, W + 1)
			if (j < w[i])
				dp[i][j] = dp[i - 1][j];
			else
				dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i]] + v[i]);
		cout << dp[n][W];
	}
}

namespace p244 {

	const int MAXN = 101;

	int n, k, a[MAXN], b[MAXN];

	void read_case() {
		read(n, k);
		for (int i = 1; i <= n; i++)
			read(a[i]);
		for (int i = 1; i <= n; i++)
			read(b[i]);
	}

	const int INF = 1000000000;
	int w[MAXN];
	int f[MAXN][10001]; // 在前i个物品中选出若干个体积之和为j的物品所能得到的最大价值
	int g[MAXN][10001];

	void solve()
	{
		for (int i = 1; i <= n; i++)
			w[i] = a[i] - b[i] * k;

		for (int i = 1; i <= n; i++)
			for (int j = 0; j <= 10000; j++)
			{
				f[i][j] = -INF;
				g[i][j] = -INF;
			}

		f[1][0] = 0;
		g[1][0] = 0;
		if (w[1] > 0)
		{
			f[1][w[1]] = a[1];
		}
		else
		{
			g[1][-w[1]] = a[1];
		}

		for (int i = 2; i <= n; i++)
		{
			for (int j = 0; j <= 10000; j++)
			{
				f[i][j] = f[i - 1][j];
				g[i][j] = g[i - 1][j];
				if (w[i] > 0 && j >= w[i])
				{
					if (f[i - 1][j - w[i]]+a[i] > f[i][j])
					{
						f[i][j] = f[i - 1][j - w[i]];
						f[i][j] += a[i];
					}
				}
				else if (w[i] <= 0 && j >= -w[i])
				{
					if (g[i - 1][j + w[i]]+a[i] > g[i][j])
					{
						g[i][j] = g[i - 1][j + w[i]];
						g[i][j] += a[i];
					}
				}
			}
		}

		int ans = -1;
		for (int i = 0; i <= 10000; i++)
		{
			int val = f[n][i]+g[n][i]; // a最小为1所以答案是大于0的
			if (val > 0 && val > ans)
			{
				ans = val;
			}
		}

		printf("%d\n", ans);
	}

}

namespace p245_part1 {

	const int MAX_N = 100;
	int n, t[MAX_N], d[MAX_N], p[MAX_N];

	void read_case() {
		read(n);
		rep(i, 0, n)
			read(t[i], d[i], p[i]);
	}

	struct Item {
		int i, t, d, p;
	};
	bool cmp(const Item& i1, const Item& i2) {
		return i1.d < i2.d;
	}

	int f[2001];
	VI g[2001];

	void solve() {
		vector<Item> a;
		rep(i, 0, n)
			a.push_back({ i+1,t[i],d[i],p[i] });
		sort(all(a), cmp);
		f[0] = 0;
		rep(i, 0, a.size()) {
			repd(j, a[i].d-1, a[i].t-1)
				if (f[j - a[i].t] + a[i].p > f[j]) {
					f[j] = f[j - a[i].t] + a[i].p;
					g[j] = g[j - a[i].t];
					g[j].push_back(a[i].i);
				}
		}
		int ans = 0, idx = 0;
		rep(i,0,2000)
			if (f[i] > ans) {
				ans = f[i];
				idx = i;
			}
		print(f[idx]);
		print(g[idx].size());
		sort(all(g[idx]));
		print(g[idx]);
	}
}

namespace p245_part2 {
	
	const int MAX_N = 100, MAX_M = 100000;
	int n, m, a[MAX_N], c[MAX_N];

	void read_case() {
		read(n, m);
		rep(i, 1, n + 1)
			read(a[i]);
		rep(i, 1, n + 1)
			read(c[i]);
	}

	int dp[MAX_N + 1][MAX_M + 1];
	void solve() {
		memset(dp, -1, sizeof(dp));
		dp[0][0] = 0;
		rep(i, 1, n + 1) {
			rep(j, 0, m + 1) {
				if (dp[i - 1][j] >= 0)
					dp[i][j] = c[i]; // dp[i][0]=c[i]意味着拼出0无消耗
				else if (j < a[i] || dp[i][j - a[i]] <= 0) // 前i-1个数拼不出j
					dp[i][j] = -1;
				else
					dp[i][j] = dp[i][j - a[i]] - 1;
			}
		}
		int ans = 0;
		rep(i, 1, m + 1)
			if (dp[n][i] >= 0)
				ans++;
		print(ans);
	}
}

namespace p246 {

	const int MAXN = 1000001;
	const int MAXK = 1000001;
	int n, k, a[MAXN];

	void read_case() {
		read(n,k);
		for (int i = 1; i <= n; i++)
			read(a[i]);
	}

	void dfs();
	int resolve_min();
	int resolve_max();

	void solve() {
		dfs();
		printf("%d %d\n", resolve_min(), resolve_max());
	}

	int ring[MAXN], rtot;//ring[i]表示第i个环的长度
	int vis[MAXN];

	int dfs(int u, int s, int len);

	void dfs()
	{	
		for (int i = 1; i <= n; i++) // 统计所有环的长度
			if (!vis[i])
				ring[++rtot] = dfs(a[i], i, 1);
	}

	int resolve_max()
	{
		int kk, ans, odd;

		std::sort(ring + 1, ring + rtot + 1);
		kk = k; ans = 0; odd = 0; // odd表示奇环数量
		for (int i = 1; kk >= 0 && i <= rtot; i++)
		{
			int cost = ring[i] / 2; // 在一个环中尽可能地淆耗k
			cost = min(cost, kk);
			kk -= cost;
			ans += cost * 2;
			if (ring[i] % 2)
				odd++;
		}
		ans += min(kk, odd); // 比如两个长度为3的环和一个长度为4的环,k=5.此时odd=2,k=1
		return ans;
	}

	std::vector<int> w;
	void add_item(int V, int C)
	{
		for (int t = 1; C > 0; t *= 2)
		{
			int mul = min(t, C);
			w.push_back(V * mul);
			C -= mul;
		}
	}

	int f[MAXK]; // 能否从前i个环中选出若干使长度之和恰好为k

	int resolve_min()
	{
		for (int i = 1; i <= rtot;) // 将长度相等的环视为一个物品
		{
			int count = 0, j = i;
			for (; j <= rtot && ring[j] == ring[i]; j++, count++)
				;
			add_item(ring[i], count);
			i = j;
		}

		f[0] = 1;
		for (int i = 0; i < w.size(); i++)
		{
			for (int j = k; j >= w[i]; j--)
			{
				if (f[j - w[i]])
					f[j] = 1;
			}
		}

		if (!f[k])
			return k + 1;
		else
			return k;
	}

	int dfs(int u, int s, int len)
	{
		if (u == s)
			return len;
		else
		{
			vis[u] = 1;
			return dfs(a[u], s, len + 1);
		}
	}
}

namespace p251 {
	
	typedef long long ll;
	ll l, r;

	void read_case() {
		read(l, r);
	}

	ll count(ll n); // 求出1~n中的同类数
	void solve() {
		printf("%lld\n", count(r) - count(l - 1));
	}

	void to_str(ll n, int a[]) // 把n的各个位倒序拆分出来,a[0]保存位数
	{
		a[0] = 0;
		for (; n; n /= 10) // 把n转化为字符串(倒序),a[0]保存位数
			a[++a[0]] = n % 10;
	}

	int a[21], sum;
	ll f[2][21][170][170];
	ll dp(bool eq, int dep, int cur_sum, int r);

	ll count(ll n)
	{
		ll ret = 0;
		memset(a, 0, sizeof(a));
		to_str(n, a);
		for (int i = 1; i <= a[0] * 9; i++) // 枚举数字之和
		{
			sum = i;
			memset(f, -1, sizeof(f)); // 使用0xff将talbe初始化为-1
			ret += dp(1, a[0], 0, 0);
		}
		return ret;
	}

	/*
	eq:已枚举出的位是否与原位相同,用来确定当前位可以枚举到几
	k:当前要枚举的位,相当于已枚举了几位,所以它应算做状态参数
	s:已枚举出的位之和
	r:已枚举出的位之和模sum的余数

	假设R是124Kxx
	如果已枚举出的位是123xxx,那么下一位可以枚举到9
	如果已枚举出的位是124xxx,那么下一位只能枚举到K

	下列程序可以枚举1~254之间的所有数字,注意254是倒序存放的
	int a[] = { 0,4,5,2,0 };
	void gen(bool eq, int k, int num)
	{
		if (k == 0) {
			print(num);
			return;
		}
		int ed = (eq ? a[k] : 9);
		for (int i = 0; i <= ed; i++) {
			gen(eq && (i==ed), k - 1, (num*10+i));
		}
	}
	gen(true, 3, 0);
	*/
	
	ll dp(bool eq, int k, int s, int r) 
	{
		if (f[eq][k][s][r] >= 0) // 只!eq && f[k][s][r]>=0也可以？？？
			return f[eq][k][s][r];

		ll tot = 0;
		if (k == 0) {
			if (s == sum && r == 0)
				tot = 1;
		}
		else {
			int ed = (eq ? a[k] : 9);
			for (int i = 0; i <= ed; i++) {
				tot += dp(eq && (i == ed), k - 1, s + i, (r * 10 + i) % sum);
			}
		}
		f[eq][k][s][r] = tot;
		return tot;
	}
}

namespace p252 {

	typedef long long ll;
	ll L, R;

	void read_case() {
		read(L,R);
	}

	ll count(ll n); // 找出1~n中的同类数

	void solve() {
		printf("%lld\n", count(R) - count(L - 1));
	}

	int a[21];
	ll f[2][21][21][2000];
	ll dp(bool eq, int dep, int cur_sum, int r);
	
	void to_str(ll n, int a[]) // 把n的各个位倒序拆分出来,a[0]保存位数
	{
		a[0] = 0;
		for (; n; n /= 10) // 把n转化为字符串(倒序),a[0]保存位数
			a[++a[0]] = n % 10;
	}

	ll count(ll n)
	{
		ll ret = 0;
		memset(a, 0, sizeof(a));
		to_str(n, a);
		for (int i = 1; i <= a[0]; i++) // 枚举支点位置
		{
			memset(f, -1, sizeof(f)); // 使用0xff将talbe初始化为-1
			ret += dp(1, a[0], i, 0);
		}
		return ret - a[0] + 1; // 0会在for循环中被计算多次,由于一个数字只有一个支点,所以其他数字不会被多次计算
	}

	// k:当前位
	// c:支点
	// s:力矩之和
	ll dp(bool eq, int k, int c, int s)
	{
		if (f[eq][k][c][s] >= 0) // 只!eq && f[k][s][r]>=0也可以？？？
			return f[eq][k][c][s];

		ll tot = 0;
		if (k == 0) {
			if (s == 0)
				tot = 1;
		}
		else {
			int ed = (eq ? a[k] : 9);
			for (int i = 0; i <= ed; i++) {
				tot += dp(eq && (i == ed), k - 1, c, s+(k-c)*i);
			}
		}
		f[eq][k][c][s] = tot;
		return tot;
	}
}

namespace p262 {

	const int MAX_N = 201;
	const int MAX_M = 201;
	int n, m;
	int a[MAX_N], w[MAX_N]; // b[i]是第i个点的点权

	void read_case() {
		read(n, m);
		rep(i,1,n+1)
			read(a[i], w[i]);
	}

	vector<VI> G;
	int f[MAX_N][MAX_M];
	int sz[MAX_N];

	void dfs(int u, int c) {
		if (c <= 0)
			return;
		rep(i, 0, G[u].size()) {
			int v = G[u][i];
			dfs(v, c-1);
			repd(j, c, 1)
				rep(k, 1, j)
					f[u][j] = max(f[u][j], f[u][j - k] + f[v][k]);
		}
	}

	void solve() {
		G = vector<VI>(n + 1);
		rep(i, 1, n+1)
			G[a[i]].push_back(i);
		rep(i, 1, n + 1)
			f[i][1] = w[i];
		dfs(0, m + 1);
		print(f[0][m+1]);
	}
}

namespace p263 {

	/*
	一组极限数据
	1
	500000 10 1000000 1000000
	*/
	const int MAX_N = 500001, MAX_K = 11;
	int N, K, A, B;
	void read_case() {
		read(N, K, A, B);
	}
	
	vector<VI> G;
	int f[MAX_N][MAX_K]; // f[i][k]表示在子树中的到点j的距离为k的点的数量
	int g[MAX_N][MAX_K]; // g[i][k]表示不在子树中的到点j的距离为k的点的数量
	int sum[MAX_N][MAX_K]; // 帮助计算g[i][k]

	void dfs1(int u, int fa) {
		rep(i, 0, G[u].size()) {
			int v = G[u][i];
			if (v == fa) continue;
			dfs1(v, u);
			rep(j, 0, K + 1) {
				if (j > 0) f[u][j] += f[v][j - 1];
				sum[u][j] += f[v][j];
			}
		}
	}

	void dfs2(int u, int fa) {
		if (fa)
			rep(j, 1, K + 1)
				g[u][j] += g[fa][j - 1];
		rep(j, 2, K + 1)
			if (j >= 2 && fa)
				g[u][j] += sum[fa][j - 2] - f[u][j - 2];
		rep(i, 0, G[u].size()) {
			int v = G[u][i];
			dfs2(v, u);
		}
			
	}

	void solve() {
		// 建图
		G = vector<VI>(N + 1);
		f[1][0] = g[1][0] = 1;
		rep(i, 2, N + 1) {
			int fa = ((LL)A * i + B) % (i - 1) + 1;
			G[fa].push_back(i);
			f[i][0] = 1;
			g[i][0] = 1;
		}
		dfs1(1, 0);
		dfs2(1, 0);
		int ans = 0;
		rep(i, 1, N + 1) {
			LL t = 0;
			rep(j, 0, K + 1)
				t += (f[i][j] + g[i][j]);
			ans ^= (t - 1);
		}
		print(ans);
	}
}

namespace p264_part1 {

	const int MAX_N = 101, MAX_K = 201;

	int N, K;
	int W[MAX_N]; // 顶点的苹果数
	int A[MAX_N], B[MAX_N]; // (A,B)

	void read_case() {
		read(N, K);
		rep(i, 1, N + 1)
			read(W[i]);
		rep(i, 0, N - 1)
			read(A[i], B[i]);
	}

	vector<VI> G;
	int f[MAX_N][MAX_K][2];
	int vis[MAX_N];

	void dfs(int u) {
		vis[u] = 1;
		rep(i, 0, K + 1)
			f[u][i][0] = f[u][i][1] = W[u];
		rep(i, 0, G[u].size()) {
			int v = G[u][i];
			if (vis[v])
				continue;
			dfs(v);
			repd(j, K, -1)
				rep(k, 1, j + 1) {
					f[u][j][1] = max(f[u][j][1], f[u][j - k][0] + f[v][k - 1][1]);
					if(k >= 2)
						f[u][j][1] = max(f[u][j][1], f[u][j - k][1] + f[v][k - 2][0]);
					if (k % 2 == 0) {
						f[u][j][0] = max(f[u][j][0], f[u][j - k][0] + f[v][k - 2][0]);
					}
				}
					

		}
	}

	void solve() {
		G = vector<VI>(N + 1);
		rep(i, 0, N - 1) {
			G[A[i]].push_back(B[i]);
			G[B[i]].push_back(A[i]);
		}
		//rep(i, 1, N + 1)
		//	f[i][0][0] = W[i];
		dfs(1);
		int ans = 0;
		rep(i, 1, N + 1) {
			ans = max(ans, f[i][K][0]);
			ans = max(ans, f[i][K][1]);
		}
		print(ans);
	}
}

namespace p264_part2 {

	const int INF = 1000000000;
	const int MAX_N = 1001;

	int N, M;
	int A[MAX_N], B[MAX_N], W[MAX_N]; // 边(A,B)权W

	void read_case() {
		read(N, M);
		rep(i, 0, N - 1)
			read(A[i], B[i], W[i]);
	}

	struct Edge {
		int to, w;
	};
	vector<vector<Edge> > G;
	int f[MAX_N]; // f[i]表示断开i与其所有叶子节点的最小花费
	int lim;// lim是二分出的最大边权
	void dfs(int u, int fa) {
		f[u] = 0;
		int flag = 1; // u是不是叶子节点
		rep(i, 0, G[u].size()) {
			int v = G[u][i].to;
			int w = G[u][i].w;
			if (v != fa) {
				flag = 0;
				dfs(v, u);
				if (w <= lim)
					f[u] += min(f[v], w);
				else
					f[u] += f[v];
			}
		}
		if (flag) f[u] = INF;
	}

	void solve() {
		G = vector<vector<Edge> >(N + 1);
		rep(i, 0, N - 1) {
			G[A[i]].push_back({ B[i],W[i] });
			G[B[i]].push_back({ A[i],W[i] });
		}
		int lo = -1, hi = INF, ans=0;
		while (hi-lo>1) {
			lim = (lo + hi) / 2;
			dfs(1, 0);
			if (f[1] <= M) {
				ans = lim;
				hi = lim;
				//print(ans);
			}
			else {
				lo = lim;
			}
		}
		print(ans);
	}
}

namespace p272 {
	const int INF = 1000000000;
	char str[17];
	void read_case() {
		read(str);
	}
	// 检查状态s对应的字符串是否是回文串
	bool check(int s, int len) {
		char buf[20];
		int tot = 0;
		rep(i, 0, len)
			if (s >> i & 1)
				buf[tot++] = str[i];
		rep(i, 0, len / 2)
			if (buf[i] != buf[tot - i - 1])
				return 0;
		return 1;
	}
	int f[70000], flag[70000];
	void solve() {
		int len = strlen(str);
		rep(i, 1, 1 << len)
			flag[i] = check(i, len);
		rep(i, 0, 1 << len)
			f[i] = INF;
		f[0] = 0;
		rep(i, 1, 1 << len)
			for (int s = i; s; s = (s - 1) & i) // 枚举i的子集
				if (flag[s])
					f[i] = min(f[i], f[i ^ s] + 1);
		print(f[(1 << len) - 1]);
	}
}

namespace p273 {
	const int MAX_N = 31, MAX_M = 160, MAX_K = 16;
	int N, M, K; // 顶点数,边数,
	int X[MAX_M], Y[MAX_M]; // 边
	int V[MAX_K]; // V[i]是高点
	void read_case() {
		read(N, M, K);
		rep(i, 0, M)
			read(X[i], Y[i]);
		rep(i, 0, K)
			read(V[i]);
	}

	int e[MAX_N][MAX_N]; // 邻接矩阵
	int ishigh[MAX_N];
	VI low;
	int f[MAX_N][70000]; // f[i][s]表示前i个低点与高点使用状态S构成的山谷数
	void solve() {
		rep(i, 0, M)
			e[X[i]][Y[i]] = e[Y[i]][X[i]] = 1;
		rep(i, 0, K)
			ishigh[V[i]] = 1;
		rep(i, 1, N + 1) // 找出全部低点
			if (!ishigh[i])
				low.push_back(i);
		vector<PII> trans[35];//trans[i]记录低点i与哪几组高点相连
		rep(i, 0, low.size())
		{
			rep(p, 0, K)
				if (e[low[i]][V[p]])
					rep(q, p + 1, K)
					if (e[low[i]][V[q]])
						trans[i].push_back({ p,q });
		}
		rep(i, 0, low.size()) {
			rep(s, 0, 1 << K) {
				f[i + 1][s] = max(f[i + 1][s], f[i][s]);
				rep(j, 0, trans[i].size()) {
					int p = trans[i][j].first;
					int q = trans[i][j].second;
					if (s >> (p - 1) & 1) continue;
					if (s >> (q - 1) & 1) continue;
					// 如果p,q不在高点集合里,那么使这两个高点与low[i]形成山谷
					int s1 = s | (1 << (p - 1)) | (1 << (q - 1));
					f[i + 1][s1] = max(f[i + 1][s1], f[i][s] + 1);
				}
			}
		}
		int ans = 0;
		rep(i, 0, 1 << K)
			ans = max(ans, f[low.size()][i]);
		print(ans);
	}
}

namespace p281 {
/*
2 1 1
aa
a
*/
	const int MAX_N = 201, MOD = 1000000007;
	int n, m, K;
	char a[MAX_N], b[MAX_N];
	void read_case() {
		read(n, m, K);
		read(a+1);
		read(b+1);
	}
	// f[i][j][k][p]表示从A[1...i]选出k个子串拼出B[1...j]的方案数,p表示A[i]是否在第k个子串中
	int f[MAX_N][MAX_N][MAX_N][2];
	void solve() {
		f[0][0][0][0]  = 1;
		rep(I, 1, n + 1) {
			int i = I;
			//memset(f[i], 0, sizeof(f[i]));
			//f[i][0][0][0]  = 1;
			rep(j, 0, m+1) {
				rep(k, 0, K + 1) {
					//if (f[i - 1][j][k][0] || f[i - 1][j][k][1])
					if (f[i - 1][j][k][0] || f[i - 1][j][k][1])
						if (a[i] == b[j + 1]) {
							f[i][j + 1][k][1] += f[i - 1][j][k][1];
							print("set1", i, j + 1, k, 1, "to", f[i][j + 1][k][1]);
							f[i][j + 1][k+1][1] += f[i - 1][j][k][0];
							f[i][j + 1][k+1][1] += f[i - 1][j][k][1];
							print("set2", i, j + 1, k+1, 1, "to", f[i][j + 1][k + 1][1]);
						}
						else {
							f[i][j][k][0] += f[i - 1][j][k][1];
							f[i][j][k][0] += f[i - 1][j][k][0];
							print("set3", i,j,k,0, "to", f[i][j][k][0]);
						}
							
				}
			}
		}
		print(f[n][m][K][0] + f[n][m][K][1] % MOD);
	}
}

namespace p282 {
	int n;
	void read_case() {
		read(n);
	}

	typedef std::vector<long long> vec;
	typedef std::vector<vec> MATRIX;
	MATRIX operator*(const MATRIX& a, const MATRIX& b);

	MATRIX pow(MATRIX a, long long b)
	{
		MATRIX operator*(const MATRIX & a, const MATRIX & b);
		MATRIX c(a.size(), vec(a.size()));
		for (int i = 0; i < a.size(); i++)
			c[i][i] = 1;
		while (b)
		{
			if (b & 1)
				c = c * a;
			b >>= 1;
			a = a * a;
		}
		return c;
	}

	const int MOD = 10007;
	MATRIX operator*(const MATRIX& a, const MATRIX& b)
	{
		MATRIX t(a.size(), vec(b[0].size()));
		for (int i = 0; i < a.size(); i++)
			for (int j = 0; j < b[0].size(); j++)
				for (int k = 0; k < b.size(); k++)
					t[i][j] = (t[i][j] + a[i][k] * b[k][j]) % MOD;
		return t;
	}

	void solve() {
		if (n == 1)
			printf("2\n");
		else
		{
			MATRIX A(4, vec(4));

			A[1][1] = 2; A[1][2] = 1; A[1][3] = 0;
			A[2][1] = 2; A[2][2] = 2; A[2][3] = 2;
			A[3][1] = 0; A[3][2] = 1; A[3][3] = 2;

			MATRIX t = pow(A, n);
			printf("%-4d\n", t[1][1]);
		}
	}
}

namespace p283 {

/*
一组简单的测试数据

2 1
1 2
*/
	const int MAX_N = 10001,MAX_C=101;
	int n, c, h[MAX_N];
	void read_case() {
		read(n, c);
		rep(i, 1, n + 1)
			read(h[i]);
	}
	int f[MAX_N][MAX_C];
	int p[MAX_N][MAX_C], qq[MAX_N][MAX_C]; // 用q会莫名其妙地出绿线
	inline int squ(int v) {
		return v * v;
	}
	const int INF = 1000000000;

	/*void solve() {
		rep(i, 1, n + 1)
			rep(j, 1, MAX_C)
			f[i][j] = p[i][j] = q[i][j] = INF;
		f[1][h[1]] = 0;
		rep(j, h[1], MAX_C)
			f[1][j] = squ(j - h[1]);
		rep(i, 2, n + 1) {
			rep(j, h[i], MAX_C) {
				rep(k, 1, MAX_C) {
					int v = f[i - 1][k] + c * abs(j - k) + squ(j - h[i]);
					if (v < f[i][j])
					{
						print("setf", i, j, "to", v);
						f[i][j] = v;
					}
				}
			}
		}
		int ans = INF;
		rep(j, h[n], MAX_C)
			ans = min(ans, f[n][j]);
		print(ans);
	}*/

	void solve() {

		rep(i, 0, n + 1)
			rep(j, 0, MAX_C)
				f[i][j] = p[i][j] = qq[i][j] = INF;
		f[1][h[1]] = 0;
		rep(j, h[1], MAX_C)
			f[1][j] = squ(j - h[1]);
		rep(j, 1, MAX_C)
			p[1][j] = min(p[1][j - 1], f[1][j] - c * j);
		repd(j, MAX_C - 1, 0)
			qq[1][j] = min(qq[1][j + 1], f[1][j] + c * j);
		rep(i, 2, n + 1) {
			rep(j, h[i], MAX_C) {
				// 之前f[i][j]的值是通过rep(k,1,MAX_C)求出的
				f[i][j] = min(f[i][j], p[i - 1][j] + c * j + squ(j - h[i])); // 瞬间求出当k<=j时的f[i][j]
				f[i][j] = min(f[i][j], qq[i - 1][j] - c * j + squ(j - h[i])); // 瞬间求出当k>=j时的f[i][j]
			}
			rep(j, 1, MAX_C) { // 仿照初始化阶段,更新p数组
				p[i][j] = min(p[i][j - 1], f[i][j] - c * j);
			}
			repd(j, MAX_C - 1, 0) { // 更新q数组
				qq[i][j] = min(qq[i][j + 1], f[i][j] + c * j);
			}
		}
		int ans = INF;
		rep(j, 1, MAX_C)
			ans = min(ans, f[n][j]);
		print(ans);
	}
}

namespace p291 {
	const int MAX_N = 500001;
	int n, m, a[MAX_N];

	void read_case() {
		n = 0;
		read(n, m);
		rep(i, 1, n+1)
			read(a[i]);
	}

	LL sum[MAX_N], f[MAX_N];
	const double INF = 1e30;
	double slope(int i, int j) { // i>j
		LL Xi,Yi,Xj,Yj;
		Xi = sum[i];
		Yi = f[i] + sum[i]*sum[i];
		Xj = sum[j];
		Yj = f[j] + sum[j] * sum[j];
		LL dY = Yi - Yj;
		LL dX = Xi - Xj;
		if (dX == 0) {
			if (dY >= 0)
				return INF;
			else
				return -INF;
		}
		else
			return (double)(Yi - Yj) / (double)(Xi - Xj);
	}

	void solve() {
		rep(i, 1, n + 1)
			sum[i] = sum[i - 1] + a[i];
		deque<int> q;
		q.push_back(0); // 对于f[1]，k只有0
		rep(i, 1, n + 1) {
			while (q.size() >= 2) {
				int I = *q.rbegin();
				int J = *(q.rbegin() + 1);
				if (slope(I, J) >= 2 * sum[i]) // 找到直线与凸壳的第一个交点
					q.pop_back();
				else
					break;
			}
			int k = q.back();
			f[i] = f[k] + (sum[i] - sum[k]) * (sum[i] - sum[k]) + m;
			while (q.size() >= 2) {
				int I = *q.rbegin();
				int J = *(q.rbegin() + 1);
				if (slope(I, J) < slope(i, I)) // 确保斜率递增
					break;
				else
					q.pop_back();
			}
			q.push_back(i);
		}
		print(f[n]);
	}
}

namespace p2A1 {
	const int MAX_N = 500001, MAX_K = 101;
	int n, k, p, a[MAX_N];
	void read_case() {
		read(n, k, p);
		rep(i, 1, n + 1)
			read(a[i]);
	}

	const int INF = 100000000;

	// 单点修改,求区间最小值
	struct Interval {
		int l, r;
		int minv; // 该线段维护的和
		Interval* lc, * rc;

		Interval() { }
		Interval(int l_, int r_)
		{
			l = l_; r = r_;
			lc = rc = NULL;
			if (l == r) {
				minv = INF;
				return;
			}
			int mid = (l + r) / 2;
			lc = new Interval(l, mid);
			rc = new Interval(mid + 1, r);
			update();
		}

		void update() {
			minv = min(lc->minv, rc->minv);
		}

		void change(int p, int v) {
			if (p<l || p>r) // p在该线段外
				return;
			if (p == l && p == r) { // 找到了点
				minv = v;
				return;
			}
			lc->change(p, v);
			rc->change(p, v);
			update();
		}

		int query(int A, int B)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return INF;
			if (l >= A && r <= B) // 该线段在[A,B]中
				return minv;
			return min(lc->query(A, B), rc->query(A, B));
		}
	};

	
	int sum[MAX_N];
	int f[MAX_N][MAX_K];
	void solve() {
		rep(i, 1, n + 1)
			sum[i] = (sum[i - 1] + a[i]) % p;
		rep(i, 1, n + 1)
			rep(j, 1, k + 1)
				f[i][j] = INF;

		Interval interval[2][MAX_N]; // interval[j]维护min{f[k][j-1] + ... |k<i},用于f[i][j]
		rep(i, 0, 2)
			rep(j, 1, k + 1)
				interval[i][j] = Interval(0, 100);

		// 首项
		f[1][0] = 0;
		f[1][a[1]] = a[1];
		rep(j, 1, k + 1) {
			interval[0][j].change(sum[1], f[1][j - 1] - sum[1]);
			interval[1][j].change(sum[1], f[1][j - 1] - sum[1] + p);
		}
			
		rep(i, 1, n + 1) {
			rep(j, 1, k + 1) {
				f[i][j] = min(f[i][j], interval[0][j].query(0, sum[i]));
				f[i][j] = min(f[i][j], interval[1][j].query(sum[i] + 1, p)) + sum[i];
			}
			rep(j, 1, k + 1) { // 仿照首项进行更新
				interval[0][j].change(sum[i], f[i][j - 1] - sum[i]);
				interval[1][j].change(sum[i], f[i][j - 1] - sum[i] + p);
			}
		}
		print(f[n][k]);
	}
}

namespace p325 {
	void read_case() {

	}

	const int INF = 100000000;
	struct Graph {
		struct Edge
		{
			int to, w;
		};
		typedef vector<Edge> Edges;
		vector<Edges> G;
		int sz; VI dist;

		Graph() { sz = 0; }
		void add_edge(int u, int v, int w) {
			G[u].push_back({ v,w });
			sz = max(sz, max(u, v));
		}

		void dijkstra(int u0) {
			dist = VI(sz + 1, INF);
			dist[u0] = 0;
			set<PII> q;
			rep(i, 1, sz + 1)
				q.insert({ dist[i],i });

			while (!q.empty())
			{
				int d = q.begin()->first;
				int u = q.begin()->second;
				q.erase(q.begin());
				if (d > dist[u]) continue;
				rep(i, 0, G[u].size())
				{
					Edge e = G[u][i];
					if (d + e.w < dist[e.to])
					{
						dist[e.to] = d + e.w;
						q.insert({ dist[e.to], e.to });
					}
				}
			}
		}
	};

	void solve() {

	}
}

namespace p326 {
	
	const int MAXN = 110; // 最大点数
	const int MAXM = 10010; // 最大边数

	int K, n, m;
	int u[MAXM], v[MAXM], w[MAXM], t[MAXM];

	struct Graph {
		struct Edge
		{
			int to, w, t;
		};
		typedef vector<Edge> Edges;
		vector<Edges> G;

		Graph() {
			G = vector<Edges>(n);
		}
		void add_edge(int u, int v, int w, int t) {
			G[u].push_back({ v,w,t });
		}

		typedef pair<int, PII> Item;
		int dijkstra(int u0) {
			priority_queue<Item, vector<Item>, greater<Item> > heap;
			int ans = -1;
			heap.push({0, {0,u0}});
			while (!heap.empty())
			{
				int d = heap.top().first;
				int c = heap.top().second.first;
				int u = heap.top().second.second;
				if (u == n)
				{
					ans = d;
					break;
				}
				heap.pop();
				rep(i,0,G[u].size())
				{
					Edge e = G[u][i];
					if (c + e.t <= K)
					{
						heap.push({ d + e.w, {c + e.t, e.to} });
					}
				}
			}
			return ans;
		}
	};

	void read_case() {
		read(K, n, m);
		rep(i,0,m)
			read(u[i], v[i], w[i], t[i]);
	}

	void solve() {
		Graph g;
		rep(i, 0, m)
			g.add_edge(u[i], v[i], w[i], t[i]);
		print(g.dijkstra(1));
	}
}

namespace p333 {
	void read_case() {

	}

	const int INF = 100000000;
	struct Graph {
		struct Edge
		{
			int from, to, w;
		};
		vector<Edge> edges;
		int sz; VI dist;

		Graph() { sz = 0; }

		void add_edge(int u, int v, int w) {
			edges.push_back({ u,v,w });
			sz = max(sz, max(u, v));
		}

		void bellman_ford(int u0) {
			dist = VI(sz + 1, INF);
			dist[u0] = 0;
			while (true)
			{
				int updated = 0;
				rep(i, 0, edges.size())
				{
					Edge& e = edges[i];
					if (dist[e.from] + e.w < dist[e.to])
					{
						dist[e.to] = dist[e.from] + e.w;
						updated = 1;
					}
				}
				if (!updated)
					break;
			}
		}
	};

	void solve() {

	}
}

namespace p334 {

	const int MAX_N = 10000;

	int n;
	double X[MAX_N], Y1[MAX_N], Y2[MAX_N], Y3[MAX_N], Y4[MAX_N];

	void read_case() {
		read(n);
		for (int i = 1; i <= n; i++)
			read(X[i], Y1[i], Y2[i], Y3[i], Y4[i]);
	}

	struct Point
	{
		double x, y;
	};

	struct Line
	{
		double x1, y1;
		double x2, y2;
	};

	struct Wall
	{
		double x;
		Line lines[4];
	};

	const int INF = 100000000;
	const double EPS = 1e-6;
	struct Graph {
		struct Edge
		{
			int from, to;
			double w;
		};
		vector<Edge> edges;
		vector<double> dist;

		Graph() {}
		Graph(int sz) { // 点的数量
			edges = vector<Edge>();
			dist = vector<double>(sz + 1, INF);
		}
		void add_edge(int u, int v, double w) {
			edges.push_back({ u,v,w });
		}

		void bellman_ford(int u0) {
			dist[u0] = 0;
			while (true)
			{
				int updated = 0;
				rep(i, 0, edges.size())
				{
					Edge& e = edges[i];
					if (!(abs(dist[e.from] - INF) < EPS) && dist[e.from] + e.w < dist[e.to]) // 判断是否是INF的方法跟整数不一样
					{
						dist[e.to] = dist[e.from] + e.w;
						updated = 1;
					}
				}
				if (!updated)
					break;
			}
		}
	};

	vector<Point> points;
	Wall walls[MAX_N];
	Graph g;

	int add_point(double x, double y)
	{
		points.push_back({ x,y });
		return points.size() - 1;
	}

	double calc_dist(Point* pt1, Point* pt2)
	{
		return sqrt(pow(pt2->x - pt1->x,2) + pow(pt2->y - pt1->y,2));
	}

	int check(Point* pt1, Wall* wall, Point* pt2) // 返回1则表示线段与某一堵墙相交
	{
		// y=mx+b
		double m = (pt2->y - pt1->y) / (pt2->x - pt1->x);
		double b = pt1->y - m * pt1->x;
		double y = m * wall->x + b;// 代入墙的x坐标得交点得y坐标
		for (int i = 1; i <= 3; i++)
		{
			if (y >= wall->lines[i].y1 && y <= wall->lines[i].y2)
				return 1;
		}
		return 0;
	}

	void link(int p1, int p2)
	{
		Point* pt1 = &points[p1];
		Point* pt2 = &points[p2];

		if (pt2->x <= pt1->x) // 防止同一堵墙上的点相连
			return;
		for (int i = 1; i <= n; i++)
		{
			if (walls[i].x > pt1->x && walls[i].x < pt2->x) // 如果一堵墙在两点之间
			{
				if (check(pt1, &walls[i], pt2)) // 判断两点的连线是否被该墙阻断
					return;
			}
		}
		g.add_edge(p1, p2, calc_dist(pt1, pt2));
	}

	void solve() {
		int st = add_point(0, 5);
		for (int i = 1; i <= n; i++)
		{
			double x = X[i], y1 = Y1[i], y2 = Y2[i], y3 = Y3[i], y4 = Y4[i];
			walls[i].x = x;

			walls[i].lines[1] = { x, 0, x, y1};
			walls[i].lines[2] = { x, y2, x, y3};
			walls[i].lines[3] = { x, y4, x, 10};

			add_point(x, y1);
			add_point(x, y2);
			add_point(x, y3);
			add_point(x, y4);
		}
		int ed = add_point(10, 5);
		g = Graph(ed);
		for (int i = st; i <= ed; i++)
			for (int j = i + 1; j <= ed; j++)
				link(i, j);
		g.bellman_ford(st);
		printf("%.2f\n", g.dist[ed]);
	}
}

namespace p335 {
	const int MAX_N = 31;
	int n,m; // n:货币数量 m:汇率表长度
	string names[MAX_N];
	string a[MAX_N], b[MAX_N]; // 货币A 兑换率 货币B
	double w[MAX_N]; // 兑换率
	void read_case() {
		read(n);
		rep(i, 1, n + 1)
			read(names[i]);
		if (n == 0)return;
		read(m);
		rep(i, 0, m)
			read(a[i], w[i], b[i]);
	}

	struct Graph { // Graph只是简单将图算法包裹在一起,所以里面直接使用全局变量
		struct Edge
		{
			int from, to;
			double w;
		};
		vector<Edge> edges;
		vector<double> dist;

		void add_edge(int u, int v, double w) {
			edges.push_back({ u,v,w });
		}

		void bellman_ford(int u0) {
			dist = vector<double>(n + 1, 0);
			dist[u0] = 1;
			rep(i,1,n+1)
			{
				rep(i, 0, edges.size())
				{
					Edge& e = edges[i];
					if (dist[e.from] * e.w > dist[e.to]) // 判断是否是INF的方法跟整数不一样
					{
						dist[e.to] = dist[e.from] * e.w;
					}
				}
			}
		}
	};

	void solve(int case_id) {
		map<string, int> id;
		rep(i, 1, n + 1)
			id[names[i]] = i;
		Graph g;
		rep(i, 0, m)
			g.add_edge(id[a[i]], id[b[i]], w[i]);
		int ans = 0;
		rep(i, 1, n + 1) {
			g.bellman_ford(i);
			if (g.dist[i] > 1) {
				printf("Cast %d: Yes", case_id);
				return;
			}
		}
		printf("Cast %d: No", case_id);
	}
}

namespace p343 {
	void read_case() {

	}

	const int INF = 100000000;
	struct Graph { // Graph只是简单将图算法包裹在一起,所以里面直接使用全局变量
		struct Edge
		{
			int from, to, w;
		};
		vector<Edge> edges;
		int sz; VI d, inq;

		Graph() { sz = 0; }

		void add_edge(int u, int v, int w) {
			edges.push_back({ u,v,w });
			sz = max(sz, max(u, v));
		}

		int spfa(int s)
		{
			d = VI(sz + 1, INF);
			inq = VI(sz + 1);
			queue<int> q;
			rep(i, 1, sz + 1) {
				d[i] = INF;
				inq[i] = 0;
			}
			d[s] = 0;
			q.push(s);
			inq[s] = 1;
			while (!q.empty())
			{
				int v = q.front();
				q.pop();
				inq[v] = 0;
				rep(i, 0, edges.size())
				{
					Edge& e = edges[i];
					if (d[e.from] + e.w < d[e.to])
					{
						d[e.to] = d[e.from] + e.w;
						if (!inq[e.to])
						{
							q.push(e.to);
							inq[e.to] = 1;
						}
					}
				}
			}
			return 0;
		}
	};

	void solve() {

	}
}

namespace p344 {
	const int MAX_V = 50001, MAX_E = 50001;
	int v, e;//点数,边数
	int w[MAX_V];//点权
	int a[MAX_E], b[MAX_E], c[MAX_E];//u,v,w
	void read_case() {
		read(v, e);
		rep(i, 1, v + 1)
			read(w[i]);
		rep(i, 1, e + 1)
			read(a[i], b[i], c[i]);
	}

	const int INF = 100000000;
	struct Graph { // Graph只是简单将图算法包裹在一起,所以里面直接使用全局变量
		struct Edge
		{
			int from, to, w;
		};
		vector<Edge> edges;
		int d[MAX_V];
		int inq[MAX_V];

		void add_edge(int u, int v, int w) {
			edges.push_back({ u,v,w });
			edges.push_back({ v,u,w });
		}

		int spfa(int s)
		{
			deque<int> q;
			rep(i, 1, v + 1) {
				d[i] = INF;
				inq[i] = 0;
			}
			d[s] = 0;
			q.push_back(s);
			inq[s] = 1;
			while (!q.empty())
			{
				int v = q[0];
				q.pop_front();
				inq[v] = 0;
				rep(i,0,edges.size())
				{
					Edge& e = edges[i];
					if (d[e.from] + e.w < d[e.to])
					{
						d[e.to] = d[e.from] + e.w;
						if (!inq[e.to])
						{
							q.push_back(e.to);
							inq[e.to] = 1;
						}
					}
				}
			}
			return 0;
		}
	};

	void solve() {
		Graph g;
		rep(i, 1, e+1)
			g.add_edge(a[i], b[i], c[i]);
		g.spfa(1);
		int ans = 0;
		rep(i, 1, v + 1)
			ans += (g.d[i] * w[i]);
		print(ans);
	}
}

namespace p345 {

	const int INF = 100000000, MAX_N=100001, MAX_M=100001;

	int n, m, c;
	int a[MAX_N]; //点所在的层
	int u[MAX_M], v[MAX_M], w[MAX_M];
	int maxl; // 最大层

	void read_case() {
		read(n, m, c); //c是层间移动代价

		maxl = 0; 
		for (int i = 1; i <= n; i++) //点所在的层
		{
			read(a[i]);
			maxl = max(a[i], maxl);
		}

		for (int i = 1; i <= m; i++)
			read(u[i], v[i], w[i]);
	}

	struct Graph { // Graph只是简单将图算法包裹在一起,所以里面直接使用全局变量
		struct Edge
		{
			int from, to, w;
		};
		vector<Edge> edges;
		int sz; // 最大的顶点编号,即顶点数量
		VI d, inq;

		Graph() {
			sz = 0;
		}
		void add_edge(int u, int v, int w) {
			edges.push_back({ u,v,w });
			sz = max(sz, max(u, v));
		}

		int spfa(int s)
		{
			queue<int> q;
			d = VI(sz + 1);
			inq = VI(sz + 1);
			rep(i, 1, sz + 1) {
				d[i] = INF;
				inq[i] = 0;
			}
			d[s] = 0;
			q.push(s);
			inq[s] = 1;
			while (!q.empty())
			{
				int v = q.front();
				q.pop();
				inq[v] = 0;
				rep(i, 0, edges.size())
				{
					Edge& e = edges[i];
					if (d[e.from] + e.w < d[e.to])
					{
						d[e.to] = d[e.from] + e.w;
						if (!inq[e.to])
						{
							q.push(e.to);
							inq[e.to] = 1;
						}
					}
				}
			}
			return 0;
		}
	};

	void solve(int case_id) {
		Graph g;

		rep(i, 1, m + 1) {
			g.add_edge(u[i], v[i], w[i]);
			g.add_edge(v[i], u[i], w[i]);
		}
			
		rep(i,1,n+1) // 虚拟顶点n+1~n+maxl,每个点代表每一层
		{
			g.add_edge(i, n + a[i], 0); // 每个点向代表点连边
			g.add_edge(n + a[i], i, 0);
		}

		rep(i, 1, maxl)
		{
			g.add_edge(n + i, n + i + 1, c); // 代表点之间连边
			g.add_edge(n + i + 1, n + i, c);
		}

		g.spfa(1);
		if (g.d[n] == INF)
		{
			printf("Case #%d: -1\n", case_id);
		}
		else
		{
			printf("Case #%d: %d\n", case_id, g.d[n]);
		}
	}
}

namespace p353 {

/*
5 5
4 3
4 2
3 2
1 2
2 5
*/
	const int MAX_N = 100, MAX_M = 4500;
	int n, m;
	int a[MAX_M], b[MAX_M];
	void read_case() {
		read(n, m);
		rep(i, 0, m)
			read(a[i], b[i]);
	}

	struct Graph { // Graph只是简单将图算法包裹在一起,所以里面直接使用全局变量
		int d[MAX_N][MAX_N];

		Graph() {
			memset(d, 0, sizeof(d));
		}
		void add_edge(int u, int v) {
			d[u][v] = 1;
		}
		void floyd()
		{
			rep(i, 1, n + 1)
				rep(j, 1, n + 1)
				rep(k, 1, n + 1)
					d[i][j] = d[i][j] || (d[i][k] && d[k][j]);
		}
	};

	void solve() {
		Graph g;
		rep(i, 1, n + 1)
			g.add_edge(i, i);
		rep(i, 0, m) {
			g.add_edge(a[i], b[i]);
			//g.add_edge(b[i], a[i]);
		}
		g.floyd();
		int ans = 0;
		rep(v, 1, n + 1) {
			int c1 = 0; // 看看有几个点能走到v
			int c2 = 0; // 看看v能走到哪几个点
			rep(u, 1, n + 1) {
				if (u == v) continue;
				if (g.d[u][v])
					c1++;
				if (g.d[v][u])
					c2++;
			}
			if (c1 + c2 == n - 1)
				ans++;
		}
		print(ans);
	}
}

namespace p362 {

/*
3 2
1000 2000 1000
1 2 1100
2 3 1300
*/
	const int MAX_N = 1000, MAX_M = 10000;
	int n, m, c[MAX_N]; // c[i]表示第i个大营的容量
	int a[MAX_N], b[MAX_N], w[MAX_N]; // a,b,w表示第ai个大营到第bi个大营至少有wi个士兵
	void read_case() {
		read(n, m);
		rep(i, 1, n + 1)
			read(c[i]);
		rep(i, 0, m)
			read(a[i], b[i], w[i]);
	}

	const int INF = 100000000;
	struct Graph { // Graph只是简单将图算法包裹在一起,所以里面直接使用全局变量
		struct Edge
		{
			int from, to, w;
		};
		vector<Edge> edges;
		int sz;
		VI d,inq;

		Graph() { sz = 0; }

		void add_edge(int u, int v, int w) {
			edges.push_back({ u,v,w });
			sz = max(sz, max(u, v));
		}

		int spfa(int s)
		{
			deque<int> q;
			d = VI(sz + 1);
			inq = VI(sz + 1);
			rep(i, 0, sz + 1) {
				d[i] = INF;
				inq[i] = 0;
			}
			d[s] = 0;
			q.push_back(s);
			inq[s] = 1;
			while (!q.empty())
			{
				int v = q[0];
				q.pop_front();
				inq[v] = 0;
				rep(i, 0, edges.size())
				{
					Edge& e = edges[i];
					if (d[e.from] + e.w < d[e.to])
					{
						d[e.to] = d[e.from] + e.w;
						if (!inq[e.to])
						{
							q.push_back(e.to);
							inq[e.to] = 1;
						}
					}
				}
			}
			return 0;
		}
	};

	int S[MAX_N];
	void solve() {
		rep(i, 1, n + 1)
			S[i] = c[i] + S[i - 1];
		Graph g;
		// 第a[i]到第b[i]个兵营至少有w[i]人
		rep(i, 0, m) {
			// S[b[i]] - S[a[i]-1] >= w[i]等价于 S[a[i]-1]-S[b[i]] <= -w[i]
			// 对于S[i] - S[j] <= c,由j向i连一条权值为c的边
			g.add_edge(b[i], a[i]-1, -w[i]);
		}
		// 第a[i]到第b[i]个兵营的人数不能超过总容量
		rep(i, 0, m) {
			// S[b[i]] - S[a[i]-1] <= w[i]
			g.add_edge(a[i]-1, b[i], w[i]);
		}
		// 每个兵营的人数不能超过其容量
		rep(i, 1, n+1) {
			// S[i] - S[i-1] <= c[i]
			g.add_edge(i - 1, i, c[i]);
		}
		// 每个兵营的人数至少为0
		rep(i, 1, n + 1) {
			// S[i] - S[i-1]>=0等价于S[i-1]-S[i]<=0
			g.add_edge(i,i-1,0);
		}
		g.spfa(n);
		print(-g.d[0]);
	}
}

namespace p393 {

/*
10 11
1 2
2 3
2 5
3 4
5 4
5 6
6 7
6 8
7 8
7 9
7 10
*/
	const int MAX_M = 100;
	int n, m;
	int u[MAX_M], v[MAX_M];

	void read_case() {
		read(n, m);
		rep(i, 0, m)
			read(u[i], v[i]);
	}

	struct Graph { // Graph只是简单将图算法包裹在一起,所以里面直接使用全局变量
		vector<VI> G;

		VI vis;
		VI dfn; // dfn[i]:顶点i的dfs序
		VI back; // back[i]:从点i或i的子孙出发能到达的最小的dfn
		/*
		在一棵dfs生成树中,点u是割点的充要条件是
		1) 如果u是根,那么u至少有两个子女
		2) 如果u不是根,那么它至少有一个孩子v,使得back[v] >= dfn[u]
		这意味着当把u拿走后,以v为根的子树会被割开,此时以v为根的子树不能通过任何边连接到原树
		*/
		VI cnt; // cnt[i]:断开点i后能形成的连通分量的个数
		int ts; // 当前序号

		Graph() {
			G = vector<VI>(n+1);
			vis = dfn = back = cnt = VI(n + 1);
			ts = 0;
		}

		void link(int u, int v) {
			G[u].push_back(v);
			G[v].push_back(u);
		}

		void tarjan(int u) {
			
			vis[u] = 1;
			dfn[u] = ++ts;
			back[u] = dfn[u];
			rep(i, 0, G[u].size()) {
				int v = G[u][i];
				if (!vis[v]) {
					tarjan(v);
					back[u] = min(back[u], back[v]);
					if (back[v] >= dfn[u]) 
						cnt[u]++;  // 说明v回不到u的祖先,u是割点,v及其子树形成一个连通分量
				}
				else 
					// 说明u->v形成一条反向边
					back[u] = min(back[u], dfn[v]);
			}
		}

		void solve() {
			rep(i, 1, n + 1) {
				if (!vis[i]) {
					tarjan(i);
					cnt[i]--;
				}
			}
			rep(i, 1, n + 1)
				// 对于一个顶点,断开它形成的连通分量个数是v+1,其中v是回不到u祖先的子节点的数量
				// 但是对于根节点,断开它形成的连通分量个数是v,所以上面cnt[i]--
				print("cutting vertex ", i, " gets the number of comps to ", cnt[i] + 1); 	
		}
	};

	void solve() {
		Graph g;
		rep(i, 0, m)
			g.link(u[i], v[i]);
		g.solve();
	}
}

namespace p394 {

/*
10 11
1 2
2 3
2 5
3 4
5 4
5 6
6 7
6 8
7 8
7 9
7 10
*/
	const int MAX_M = 100;
	int n, m;
	int u[MAX_M], v[MAX_M];

	void read_case() {
		read(n, m);
		rep(i, 0, m)
			read(u[i], v[i]);
	}

	struct Graph { // Graph只是简单将图算法包裹在一起,所以里面直接使用全局变量
		vector<VI> G;

		VI vis;
		VI dfn; // dfn[i]:顶点i的dfs序
		VI back; // back[i]:从点i出发能到达的最小的dfn
		int ts; // 当前序号
		stack<PII> st; // 用于输出点双连通分量

		Graph() {
			G = vector<VI>(n + 1);
			vis = dfn = back = VI(n + 1);
			ts = 0;
		}

		void link(int u, int v) {
			G[u].push_back(v);
			G[v].push_back(u);
		}

		void tarjan(int u) {
			vis[u] = 1;
			dfn[u] = ++ts;
			back[u] = dfn[u];
			rep(i, 0, G[u].size()) {
				int v = G[u][i];
				if (!vis[v]) {
					st.push({ u,v });
					tarjan(v);
					back[u] = min(back[u], back[v]);
					if (back[v] >= dfn[u]) {
						while (1) {
							int u_ = st.top().first;
							int v_ = st.top().second;
							st.pop();
							print(u_, v_);
							if ((u_ == u && v_ == v) || (u_ == v && v_ == u))
								break;
						}
						print("");
					}
						
				}
				else
					// 说明u->v形成一条反向边
					back[u] = min(back[u], dfn[v]);
			}
		}

		void solve() {
			rep(i, 1, n + 1)
				if (!vis[i])
					tarjan(i);
		}
	};

	void solve() {
		Graph g;
		rep(i, 0, m)
			g.link(u[i], v[i]);
		g.solve();
	}
}

namespace p397 {

/*
10 11
1 2
2 3
2 5
3 4
5 4
5 6
6 7
6 8
7 8
7 9
7 10
*/
	const int MAX_M = 100;
	int n, m;
	int u[MAX_M], v[MAX_M];

	void read_case() {
		read(n, m);
		rep(i, 0, m)
			read(u[i], v[i]);
	}

	struct Graph { // Graph只是简单将图算法包裹在一起,所以里面直接使用全局变量
		vector<VI> G;

		VI vis;
		VI dfn; // dfn[i]:顶点i的dfs序
		VI back; // back[i]:从点i出发能到达的最小的dfn
		/*
		在一棵dfs生成树中,边(u,v)的充要条件是
		1) (u,v)是生成树的边
		2) 对于u的一个孩子v,有dfn[u]<low[v]
		这意味着当把边(u,v)拿走后,以v为根的子树会被割开,此时以v为根的子树不能通过任何边连接到原树
		*/
		int ts; // 当前序号

		Graph() {
			G = vector<VI>(n + 1);
			vis = dfn = back = VI(n + 1);
			ts = 0;
		}

		void link(int u, int v) {
			G[u].push_back(v);
			G[v].push_back(u);
		}

		void tarjan(int u, int fa) {

			vis[u] = 1;
			dfn[u] = ++ts;
			back[u] = dfn[u];
			rep(i, 0, G[u].size()) {
				int v = G[u][i];
				if (!vis[v]) {
					tarjan(v, u);
					back[u] = min(back[u], back[v]);
					if (back[v] > dfn[u])
						print(u, v); // (u,v)是割边
				}
				else if(v != fa)
					back[u] = min(back[u], dfn[v]);
			}
		}

		void solve() {
			rep(i, 1, n + 1) {
				if (!vis[i]) {
					tarjan(i, 0);
				}
			}
		}
	};

	void solve() {
		Graph g;
		rep(i, 0, m)
			g.link(u[i], v[i]);
		g.solve();
	}
}

namespace p398q1 {

	const int MAX_N = 1001,MAX_M = 1000001;

	int n, m;
	int u[MAX_M], v[MAX_M]; // u[i]和v[i]之间有仇恨关系

	void read_case() {
		read(n, m);
		rep(i, 0, m)
			read(u[i], v[i]);
	}

	int ans;//最终答案

	struct Component { // 一个连通分量

		int sz; // 连通分量大小
		VI inq, vis, color;
		bool found; // 是否有奇圈

		Component() {
			inq = vis = color = VI(n + 1);
			sz = 0;
		}
		void add_point(int u) {
			if (!inq[u]) {
				inq[u] = 1;
				sz++;
			}
		}
		void dfs(vector<VI> &G, int u, int c) {
			vis[u] = 1;
			color[u] = c;
			rep(i, 0, G[u].size()) {
				int v = G[u][i];
				if (!inq[v]) continue; // 只遍历分量中的点
				if (!vis[v] && !found)
					dfs(G, v, !c); // 用相反颜色染下一个点
				else
					if (color[v] == c)
						found = true;
			}
		}
		bool check(vector<VI> &G) { // 看看其中有没有奇圈
			found = false;
			rep(i, 1, n + 1)
				if (inq[i]) {
					dfs(G, i, 0);
					break;
				}
			return found;
		}
	};

	struct Graph { // Graph只是简单将图算法包裹在一起,所以里面直接使用全局变量
		vector<VI> G;

		VI vis;
		VI dfn; // dfn[i]:顶点i的dfs序
		VI back; // back[i]:从点i出发能到达的最小的dfn
		int ts; // 当前序号
		stack<PII> st; // 用于输出点双连通分量

		Graph() {
			G = vector<VI>(n + 1);
			vis = dfn = back = VI(n + 1);
			ts = 0;
		}

		void link(int u, int v) {
			G[u].push_back(v);
			G[v].push_back(u);
		}

		void tarjan(int u) {
			vis[u] = 1;
			dfn[u] = ++ts;
			back[u] = dfn[u];
			rep(i, 0, G[u].size()) {
				int v = G[u][i];
				if (!vis[v]) {
					st.push({ u,v });
					tarjan(v);
					back[u] = min(back[u], back[v]);
					if (back[v] >= dfn[u]) {
						Component comp;
						while (1) {
							int u_ = st.top().first;
							int v_ = st.top().second;
							st.pop();
							comp.add_point(u_);
							comp.add_point(v_);
							if ((u_ == u && v_ == v) || (u_ == v && v_ == u))
								break;
						}
						if (comp.check(G))
							ans += comp.sz;
						//print("");
					}

				}
				else
					// 说明u->v形成一条反向边
					back[u] = min(back[u], dfn[v]);
			}
		}

		void solve() {
			rep(i, 1, n + 1)
				if (!vis[i])
					tarjan(i);
		}
	};

	int tab[MAX_N][MAX_N];//用于建立补图
	void solve() {
		memset(tab, 0, sizeof(tab));
		rep(i, 0, m) {
			tab[u[i]][v[i]] = 1;
			tab[v[i]][u[i]] = 1;
		}
			
		ans = 0;
		Graph g;
		rep(i, 1, n + 1)
			rep(j, 1, i + 1)
			if (j != i && tab[i][j] == 0)
				g.link(i, j);
		g.solve();
		print(ans);
	}
}

namespace p523 {

	const int MAX_N = 500000;
	int n;
	VI a;

	void read_case() {
		read(n);
		a = VI(n);
		rep(i, 0, n)
			read(a[i]);
	}

	struct Interval
	{
		int l, r;
		int sum; // 该线段维护的和
		Interval* lc, * rc;

		Interval(int l_, int r_) 
		{
			l = l_; r = r_;
			sum = 0;
			lc = rc = NULL;
			if (l == r) {
				return;
			}
			int mid = (l + r) / 2;
			lc = new Interval(l, mid);
			rc = new Interval(mid+1, r);
			update();
		}
		
		void update() 
		{
			sum = lc->sum + rc->sum;
		}

		void add(int p, int v) 
		{
			if (p<l || p>r) // p在该线段外
				return;
			if (p == l && p == r) // p在该线段中
			{
				sum += v;
				return;
			}
			lc->add(p, v);
			rc->add(p, v);
			update();
		}

		int get_sum(int A, int B) 
		{
			if (l>B || r<A) // 该线段在[A,B]外
				return 0;
			if (l >= A && r <= B) // 该线段在[A,B]中
				return sum;
			return (lc->get_sum(A, B) + rc->get_sum(A, B));
		}
	};

	void solve() {
		VI b = a; // 离散化
		sort_unique(b);
		rep(i, 0, n)
			a[i] = lower_bound(all(b), a[i]) - b.begin() + 1;

		Interval* interval = new Interval(1, b.size());
		LL ans = 0;
		rep(i, 0, n) {
			ans += interval->get_sum(a[i] + 1, b.size());
			interval->add(a[i], 1);
		}
		print(ans);
	}
}

namespace p524 {
	void read_case() {

	}

	struct Interval
	{
		int l, r;
		int sz, tag, sum; // 该线段维护的和
		Interval* lc, * rc;

		Interval(VI& a, int l_, int r_)
		{
			l = l_; r = r_;
			tag = sum = 0;
			lc = rc = NULL;
			if (l == r) {
				sz = 1;
				sum = a[l];
				return;
			}
			int mid = (l + r) / 2;
			lc = new Interval(a, l, mid);
			rc = new Interval(a, mid + 1, r);
			update();
		}

		void update()
		{
			sz = lc->sz + rc->sz;
			sum = lc->sum + rc->sum;
		}

		void pass() // 标记下传
		{
			if (tag) {
				lc->tag = tag;
				lc->sum = lc->sz * tag;
				rc->tag = tag;
				rc->sum = rc->sz * tag;
				tag = 0;
			}
		}

		void change(int A, int B, int v)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return;
			if (l >= A && r <= B) // 该线段在[A,B]中
			{
				sum = sz * v;
				tag = v;
				return;
			}
			pass();
			lc->change(A, B, v);
			rc->change(A, B, v);
			update();
		}

		LL get_sum(int A, int B)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return 0;
			if (l >= A && r <= B) // 该线段在[A,B]中
				return sum;
			pass();
			return (lc->get_sum(A, B) + rc->get_sum(A, B));
		}
	};

	void solve() {

	}
}

namespace p525_part1 {

	struct Interval
	{
		int l, r;
		int sz, tag, sum; // 该线段维护的和
		Interval* lc, * rc;

		Interval(VI &a, int l_, int r_)
		{
			l = l_; r = r_;
			sz = tag = sum = 0;
			lc = rc = NULL;
			if (l == r) {
				sz = 1;
				sum = a[l];
				return;
			}
			int mid = (l + r) / 2;
			lc = new Interval(a, l, mid);
			rc = new Interval(a, mid + 1, r);
			update();
		}

		void update()
		{
			sz = lc->sz + rc->sz;
			sum = lc->sum + rc->sum;
		}

		void pass() // 标记下传
		{
			if (tag) {
				lc->tag += tag;
				lc->sum += lc->sz * tag;
				rc->tag += tag;
				rc->sum += rc->sz * tag;
				tag = 0;
			}
		}

		void add(int A, int B, int v)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return;
			if (l >= A && r <= B) // 该线段在[A,B]中
			{
				sum += sz * v;
				tag += v;
				return;
			}
			pass();
			lc->add(A, B, v);
			rc->add(A, B, v);
			update();
		}

		LL get_sum(int A, int B)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return 0;
			if (l >= A && r <= B) // 该线段在[A,B]中
				return sum;
			pass();
			return (lc->get_sum(A, B) + rc->get_sum(A, B));
		}
	};

	void read_case() {

	}

	void solve() {

	}
}

namespace p525_part2 {

	void read_case() {

	}

	const int INF = 1000000000;

	struct Interval
	{
		int l, r;
		int tag, minv; // 该线段维护的值
		Interval* lc, * rc;

		Interval(VI& a, int l_, int r_)
		{
			l = l_; r = r_;
			tag = minv = 0;
			lc = rc = NULL;
			if (l == r) {
				minv = a[l];
				return;
			}
			int mid = (l + r) / 2;
			lc = new Interval(a, l, mid);
			rc = new Interval(a, mid + 1, r);
			update();
		}

		void update()
		{
			minv = min(lc->minv, rc->minv);
		}

		void pass() // 标记下传
		{
			if (tag) {
				lc->tag += tag;
				lc->minv += tag;
				rc->tag += tag;
				rc->minv += tag;
				tag = 0;
			}
		}

		void add(int A, int B, int v)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return;
			if (l >= A && r <= B) // 该线段在[A,B]中
			{
				minv += v;
				tag += v;
				return;
			}
			pass();
			lc->add(A, B, v);
			rc->add(A, B, v);
			update();
		}

		int get_minv(int A, int B)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return INF;
			if (l >= A && r <= B) // 该线段在[A,B]中
				return minv;
			pass();
			int minv1 = lc->get_minv(A, B);
			int minv2 = rc->get_minv(A, B);
			return min(minv1, minv2);
		}
	};

	void solve() {

	}
}

namespace p525_part3 {
	void read_case() {

	}

	const int INF = 1000000000;

	struct Interval
	{
		int l, r;
		int tag, minv, cnt; // 该线段维护的值
		Interval* lc, * rc;

		Interval(VI& a, int l_, int r_)
		{
			l = l_; r = r_;
			tag = minv = 0;
			lc = rc = NULL;
			if (l == r) {
				minv = a[l];
				cnt = 1;
				return;
			}
			int mid = (l + r) / 2;
			lc = new Interval(a, l, mid);
			rc = new Interval(a, mid + 1, r);
			update();
		}

		void update()
		{
			if (lc->minv < rc->minv) {
				minv = lc->minv;
				cnt = lc->cnt;
			}
			else if (rc->minv < lc->minv) {
				minv = rc->minv;
				cnt = rc->cnt;
			}
			else if (rc->minv == lc->minv) {
				minv = lc->minv;
				cnt = lc->cnt + rc->cnt;
			}
		}

		void pass() // 标记下传
		{
			if (tag) {
				lc->tag += tag;
				lc->minv += tag;
				rc->tag += tag;
				rc->minv += tag;
				tag = 0;
			}
		}

		void add(int A, int B, int v)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return;
			if (l >= A && r <= B) // 该线段在[A,B]中
			{
				minv += v;
				tag += v;
				return;
			}
			pass();
			lc->add(A, B, v);
			rc->add(A, B, v);
			update();
		}

		PII query(int A, int B)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return make_pair(INF,0);
			if (l >= A && r <= B) // 该线段在[A,B]中
				return make_pair(minv, cnt);
			pass();
			PII p1 = lc->query(A, B);
			PII p2 = rc->query(A, B);
			if (p1.first < p2.first)
				return p1;
			else if (p2.first < p1.first)
				return p2;
			else if (p1.first == p2.first)
				return make_pair(p1.first, p1.second + p2.second);
		}
	};

	void solve() {

	}
}

namespace p532 {

	const int MAXN = 1000000;
	struct RECT { int x1, y1, x2, y2; };
	int n;
	RECT r[MAXN];

	void read_case() {
		read(n);
		for (int i = 1; i <= n; i++)
			read(r[i].x1, r[i].y1, r[i].x2, r[i].y2);
	}

	VI xbin, ybin;

	struct EVENT
	{
		int x1, x2, v;
	};
	typedef vector<EVENT> events;
	vector<events> g;

	void add_event(int y, int x1, int x2, int v)
	{
		g[y].push_back({ x1,x2,v });
	}

	void preprocess() 
	{
		for (int i = 1; i <= n; i++) // 坐标离散化
		{
			xbin.push_back(r[i].x1);
			xbin.push_back(r[i].x2);
			ybin.push_back(r[i].y1);
			ybin.push_back(r[i].y2);
		}

		sort_unique(xbin);
		sort_unique(ybin);

		g = vector<events>(ybin.size());
		for (int i = 1; i <= n; i++) // 添加事件
		{
			int x1 = lower_bound(all(xbin), r[i].x1) - xbin.begin();
			int x2 = lower_bound(all(xbin), r[i].x2) - xbin.begin();
			int y1 = lower_bound(all(ybin), r[i].y1) - ybin.begin();
			int y2 = lower_bound(all(ybin), r[i].y2) - ybin.begin();
			if (x1 < x2)
			{
				add_event(y1, x1 + 1, x2, 1);
				add_event(y2, x1 + 1, x2, -1);
			}
		}
	}

	const int INF = 1000000000;
	struct Interval // 区间修改,求区间最小值
	{
		int l, r;
		int tag, len, minv; // 最小值的数量(即未被覆盖的线段长度),最小值
		Interval* lc, * rc;

		Interval(int l_, int r_)
		{
			l = l_; r = r_;
			tag = minv = 0;
			lc = rc = NULL;
			if (l == r) {
				if (l_ - 1 >= 0)
					len = xbin[r_] - xbin[l_-1]; // [i,j]表示点i-1到点j之间线段的实际长度
				else
					len = 0;
				return;
			}
			int mid = (l + r) / 2;
			lc = new Interval(l, mid);
			rc = new Interval(mid + 1, r);
			update();
		}

		void update() // 合并节点信息
		{
			if (lc->minv == rc->minv) {
				minv = lc->minv;
				len = lc->len + rc->len;
			}
			else if (lc->minv < rc->minv) {
				minv = lc->minv;
				len = lc->len;
			}
			else {
				minv = rc->minv;
				len = rc->len;
			}
		}

		void pass() // 标记下传
		{
			if (tag) {
				lc->tag += tag;
				lc->minv += tag;
				rc->tag += tag;
				rc->minv += tag;
				tag = 0;
			}
		}

		void add(int A, int B, int v)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return;
			if (l >= A && r <= B) // 该线段在[A,B]中
			{
				minv += v;
				tag += v;
				return;
			}
			pass();
			lc->add(A, B, v);
			rc->add(A, B, v);
			update();
		}

		int get_minv(int A, int B)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return INF;
			if (l >= A && r <= B) // 该线段在[A,B]中
				return minv;
			pass();
			int minv1 = lc->get_minv(A, B);
			int minv2 = rc->get_minv(A, B);
			return min(minv1, minv2);
		}
	};

	void solve() {
		LL len, ans;

		preprocess();
		Interval interval(0, xbin.size()-1);
		
		len = ans = 0;
		rep(i,0,ybin.size())
		{
			len = interval.len; // 被覆盖的长度=总长-未被覆盖的长度
			if (interval.minv > 0)
				len = 0; // 全被覆盖
			if(i>0)
				ans += (ybin[i] - ybin[i - 1]) * (xbin[xbin.size()-1] - xbin[0] - len);
			rep(j,0,g[i].size())
				interval.add(g[i][j].x1, g[i][j].x2, g[i][j].v);
		}

		printf("%lld\n",ans);
	}
}

namespace p533 {

	const int MAXN = 30010;
	const int MAXQ = 100010;

	typedef long long ll;

	struct QUERY
	{
		int l, r, id;
		bool operator<(const QUERY& r) { // 按l排序
			return l < r.l;
		}
	};

	struct S1
	{
		int i; // 在输入数列A中的索引
		int v;
		int left; // 在输入数列A中上一个相同数字的位置

		bool operator<(const S1& r) { // 按left排序
			return left < r.left;
		}
	};

	int n, a[MAXN];
	int q;
	QUERY queries[MAXQ];

	void read_case() {
		read(n);
		for (int i = 1; i <= n; i++)
		{
			read(a[i]);
		}
		read(q);
		for (int i = 1; i <=  q; i++)
		{
			read(queries[i].l, queries[i].r);
			queries[i].id = i;
		}
	}

	int last[MAXN];
	S1 A[MAXN]; // 是输入数列的拷贝,并按left进行了排序

	void preprocess()
	{
		// 比如 a{3,8,4,7}
		// 离散化后
		// bin{3,4,7,8}
		// a{1,4,2,3}

		VI bin;
		for (int i = 1; i <= n; i++) // 对a离散化,用于找到一个数字左侧第一个相同数字的位置
			bin.push_back(a[i]);
		sort_unique(bin);
		for (int i = 1; i <= n; i++)
			a[i] = lower_bound(all(bin), a[i]) - bin.begin(); // 保存数字在离散化后对应的id,范围是1~N

		for (int i = 1; i <= n; i++)
			last[i] = 0;
		// 之后的处理均针对离散化后的值(从1~N的数列a)进行
		for (int i = 1; i <= n; i++)
		{
			A[i].i = i;
			A[i].v = bin[a[i]];
			A[i].left = last[a[i]];
			last[a[i]] = i;
		}
		sort(queries+1, queries+q+1);
		sort(A+1, A+n+1);
	}

	struct Interval
	{
		int l, r;
		int sum; // 区间和
		Interval* lc, * rc;

		Interval(int l_, int r_)
		{
			l = l_; r = r_;
			sum = 0;
			lc = rc = NULL;
			if (l == r) {
				return;
			}
			int mid = (l + r) / 2;
			lc = new Interval(l, mid);
			rc = new Interval(mid + 1, r);
			update();
		}

		void update()
		{
			sum = lc->sum + rc->sum;
		}

		void add(int p, int v)
		{
			if (p<l || p>r) // p在该线段外
				return;
			if (p == l && p == r) // p在该线段中
			{
				sum += v;
				return;
			}
			lc->add(p, v);
			rc->add(p, v);
			update();
		}

		int get_sum(int A, int B)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return 0;
			if (l >= A && r <= B) // 该线段在[A,B]中
				return sum;
			return (lc->get_sum(A, B) + rc->get_sum(A, B));
		}
	};

	LL ans[MAXQ];
	void solve()
	{
		preprocess();
		Interval interval(1, n);
		int j = 1;
		for (int i = 1; i <= q; i++) // 对询问按左端点排序
		{
			while (j <= n && A[j].left < queries[i].l) // 如果一个数字的left不在这个区间内，意味着它在这个区间中第一次出现
			{
				interval.add(A[j].i, A[j].v);
				j++;
			}
			ans[queries[i].id] = interval.get_sum(queries[i].l, queries[i].r);
		}
		for (int i = 1; i <= q; i++)
			printf("%I64d\n", ans[i]);
	}
}

namespace p534 {

	const int MAXN = 200009;

	int n, p[MAXN], v[MAXN];
	int ans[MAXN];

	void read_case() {
		scanf("%d", &n);
		for (int i = 1; i <= n; i++)
			scanf("%d %d", &p[i], &v[i]);
	}

	struct Interval
	{
		int l, r;
		int sum; // 区间和
		Interval* lc, * rc;

		Interval(int l_, int r_)
		{
			l = l_; r = r_;
			sum = 0;
			lc = rc = NULL;
			if (l == r) {
				sum = 1;
				return;
			}
			int mid = (l + r) / 2;
			lc = new Interval(l, mid);
			rc = new Interval(mid + 1, r);
			update();
		}

		void update()
		{
			sum = lc->sum + rc->sum;
		}

		int find(int k)
		{
			if (l == r)
			{
				sum = 0;
				return (l); // 返回被修改位置.注意!这里是l不是1
			}
			int mid = (l + r) / 2, ret = 0;
			if (k <= lc->sum) // 注意!这里是小于等于
				ret = lc->find(k);
			else if (k > lc->sum)
				ret = rc->find(k - lc->sum);
			update();
			return ret;
		}
	};

	void solve() {
		Interval interval(1, n);
		for (int i = n; i >= 1; i--)
		{
			int k = interval.find(p[i] + 1);
			ans[k] = v[i];
		}
		for (int i = 1; i <= n; i++)
			printf("%d ", ans[i]);
		printf("\n");
	}
}

namespace p535 {

	const int MAXN = 50009;

	int n, m;
	stack<int> last_d;

	struct Interval
	{
		int l, r;
		int lmax, rmax; // 以左端点为终点的连续地道长度,以右端点为终点的连续地道长度
		Interval* lc, * rc;

		Interval(int l_, int r_)
		{
			l = l_; r = r_;
			lmax = rmax = 1;
			lc = rc = NULL;
			if (l == r) {
				lmax = rmax = 1;
				return;
			}
			int mid = (l + r) / 2;
			lc = new Interval(l, mid);
			rc = new Interval(mid + 1, r);
			update();
		}

		void update()
		{
			int mid = (l + r) / 2;
			if (lc->lmax == mid - l + 1) // 如果左区间被地道覆盖...
				lmax = lc->lmax + rc->lmax;
			else
				lmax = lc->lmax;
			if (rc->rmax == r - mid)  // 如果右区间被地道覆盖,那么以右端点为终点的地道长度为
								// 整个右区间加上以左区间右端点为终点的地道长度
				rmax = rc->rmax + lc->rmax;
			else
				rmax = rc->rmax;
		}

		void change(int p, int v)
		{
			if (l == r)
			{
				lmax = rmax = v;
				return;
			}
			int mid = (l + r) / 2;
			if (p <= mid) // 注意!这里是小于等于
				lc->change(p, v);
			else
				rc->change(p, v);
			update();
		}

		int query(int k) {
			if (l == r)
				return lmax;
			int mid = (l + r) / 2;
			if (k <= mid)
			{
				if (mid - lc->rmax + 1 <= k) // 如果k在左区间的rmax内...
					return lc->rmax + rc->lmax;
				else
					return lc->query(k);
			}
			else if (mid + 1 <= k)
			{
				if (k <= mid + rc->lmax) // 如果k在右区间的lmax内,同上
					return lc->rmax + rc->lmax;
				else
					return rc->query(k);
			}
		}
	};

	void read_case() {
		scanf("%d %d", &n, &m);
	}

	void solve() {
		Interval interval(1, n);
		int ans = 0;
		for (int i = 1; i <= m; i++)
		{
			char op[2]; int p = 0;
			scanf("%s", &op);
			switch (op[0])
			{
			case 'Q': // 查询含第k段的最长地道
				scanf("%d", &p);
				ans = interval.query(p);
				printf("%d\n", ans);
				break;
			case 'D': // 炸毁第k段地道
				scanf("%d", &p);
				interval.change(p, 0);
				last_d.push(p);
				break;
			case 'R': // 修复上一段被炸的地道
				interval.change(last_d.top(), 1);
				last_d.pop();
				break;
			}
		}
	}
}

namespace p536 {

	const int MAXQ = 100009;

	const int INF = 100000000;
	struct Graph {
		struct Edge
		{
			int to, w;
		};
		typedef vector<Edge> Edges;
		vector<Edges> G; int sz;
		VI dist;

		Graph() { sz = 0; }

		void init(int n_) {
			sz = n_;
			G = vector<Edges>(sz + 1);
		}
		void add_edge(int u, int v, int w) {
			G[u].push_back({ v,w });
		}
		void dijkstra(int u0) {
			dist = VI(sz + 1, INF);
			dist[u0] = 0;
			set<PII> q;
			rep(i, 1, sz + 1)
				q.insert({ dist[i],i });

			while (!q.empty())
			{
				int d = q.begin()->first;
				int u = q.begin()->second;
				q.erase(q.begin());
				if (d > dist[u]) continue;
				rep(i, 0, G[u].size())
				{
					Edge e = G[u][i];
					if (d + e.w < dist[e.to])
					{
						dist[e.to] = d + e.w;
						q.insert({ dist[e.to], e.to });
					}
				}
			}
		}
	};

	int n, m, s; // 初始n个点,m个询问,起点为s
	int vtot;
	Graph g;

	struct Interval
	{
		int l, r, id;
		Interval* lc, * rc;

		Interval(int l_, int r_)
		{
			l = l_; r = r_;
			lc = rc = NULL;
			if (l == r) {
				id = l; // 叶节点对应[1,n]
				return;
			}
			vtot += 1;//非叶节点
			id = vtot;
			int mid = (l + r) / 2;
			lc = new Interval(l, mid);
			rc = new Interval(mid + 1, r);
		}

		void build(bool rev = false) { // 建图,rev表示是否反向连边
			if (l == r)
				return;
			if (!rev) {
				g.add_edge(id, lc->id, 0);
				g.add_edge(id, rc->id, 0);
			}
			else {
				g.add_edge(lc->id, id, 0);
				g.add_edge(rc->id, id, 0);
			}
			lc->build(rev);
			rc->build(rev);
		}

		void add_edge(int u, int A, int B, int w, bool rev = false) {
			if (l >= A && r <= B) { // 在[A,B]中
				if (!rev)
					g.add_edge(u, id, w);
				else
					g.add_edge(id, u, w);
			}
			else if (r < A || l > B) // 与[A,B]不相交
				return;
			else {
				lc->add_edge(u, A, B, w, rev);
				rc->add_edge(u, A, B, w, rev);
			}
		}
	};

	void read_case() {
		
	}

	void solve() {
		scanf("%d %d %d", &n, &m, &s);

		vtot = n;
		Interval root(1, n), root1(1, n);
		g.init(vtot);
		root.build();
		root1.build(true);

		rep(i, 0, m) {
			int op, a, b, c, d;
			op = a = b = c = d = 0;
			scanf("%d", &op);
			switch (op) {
			case 1: // v to v
				scanf("%d %d %d", &a, &b, &c);
				g.add_edge(a,b,c);
				break;
			case 2: // v to range
				scanf("%d %d %d %d", &a, &b, &c, &d);
				root.add_edge(a,b,c,d);
				break;
			case 3: // range to v
				scanf("%d %d %d %d", &a, &b, &c, &d);
				root1.add_edge(a, b, c, d, true);
				break;
			}
		}
		g.dijkstra(s);
		rep(i, 1, n + 1)
			print(g.dist[i]);
	}
}

namespace p537 {
	
	const int MAXN = 100009;
	int n, m;

	void read_case() {

	}

	struct Graph { // 只是对图算法的简单封装,所以直接用全局变量
		struct EDGE { int from, to; };
		typedef vector<EDGE> edges;
		vector<edges> g;
		int ts; VI st, ed;
		
		Graph() { // 将要粘贴在solve开头封装在构造函数里
			g = vector<edges>(n + 1);
			ts = 0; st = ed = VI(n + 1);
		}

		void add_edge(int u, int v) {
			g[u].push_back({ u,v });
		}

		void dfs(int u, int fa) {
			st[u] = ++ts;
			rep(i, 0, g[u].size())
			{
				EDGE& e = g[u][i];
				if (e.to != fa)
				{
					dfs(e.to, u);
				}
			}
			ed[u] = ts;
		}
	};
	
	struct Interval
	{
		int l, r;
		int sum; // 该线段维护的和
		Interval* lc, * rc;

		Interval(int l_, int r_)
		{
			l = l_; r = r_;
			sum = 0;
			lc = rc = NULL;
			if (l == r) {
				sum = 1;
				return;
			}
			int mid = (l + r) / 2;
			lc = new Interval(l, mid);
			rc = new Interval(mid + 1, r);
			update();
		}

		void update()
		{
			sum = lc->sum + rc->sum;
		}

		void change(int p)
		{
			if (p<l || p>r) // p在该线段外
				return;
			if (p == l && p == r) // p在该线段中
			{
				if (sum == 1)
					sum = 0;
				else
					sum = 1;
				return;
			}
			lc->change(p);
			rc->change(p);
			update();
		}

		int get_sum(int A, int B)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return 0;
			if (l >= A && r <= B) // 该线段在[A,B]中
				return sum;
			return (lc->get_sum(A, B) + rc->get_sum(A, B));
		}
	};

	void solve() {
		scanf("%d", &n);

		Graph g;

		for (int i = 1; i <= n - 1; i++)
		{
			int u, v;
			scanf("%d %d", &u, &v);
			g.add_edge(u, v);
			g.add_edge(v, u);
		}
		scanf("%d", &m);

		g.dfs(1, 0);
		Interval interval(1, g.ts);
		for (int i = 1; i <= m; i++)
		{
			char op[2]; int p;
			scanf("%s%d", &op, &p);
			if (op[0] == 'Q')
				printf("%d\n", interval.get_sum(g.st[p], g.ed[p]));
			else if (op[0] == 'C')
				interval.change(g.st[p]);
		}
	}
}

namespace p538 {

	const int MAXN = 100009;
	int n, m, t;//共有t种颜色

	void read_case() {

	}

	struct Interval
	{
		int l, r;
		int color, tag; // 该线段维护的和
		Interval* lc, * rc;

		Interval(int l_, int r_)
		{
			l = l_; r = r_;
			color = tag = 0;
			lc = rc = NULL;
			if (l == r) {
				color = 1;
				tag = 0;
				return;
			}
			int mid = (l + r) / 2;
			lc = new Interval(l, mid);
			rc = new Interval(mid + 1, r);
			update();
		}

		void update()
		{
			color = lc->color | rc->color;
		}

		void pass() // 标记下传
		{
			if (tag) {
				lc->tag = tag;
				lc->color = tag;
				rc->tag = tag;
				rc->color = tag;
				tag = 0;
			}
		}

		void change(int A, int B, int v)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return;
			if (l >= A && r <= B) // 该线段在[A,B]中
			{
				color = v;
				tag = v;
				return;
			}
			pass();
			lc->change(A, B, v);
			rc->change(A, B, v);
			update();
		}

		int query(int A, int B)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return 0;
			if (l >= A && r <= B) // 该线段在[A,B]中
				return color;
			pass();
			return (lc->query(A, B) | rc->query(A, B));
		}
	};

	int bitcount(unsigned int x)
	{
		int b;
		for (b = 0; x != 0; x >>= 1)
			if (x & 1)
				b++;
		return b;
	}

	void solve() {
		scanf("%d %d %d", &n, &t, &m);
		Interval interval(1, n);
		for (int i = 1; i <= m; i++)
		{
			char op[2]; int a, b, c;
			scanf("%s", &op);
			if (op[0] == 'C')
			{
				scanf("%d %d %d", &a, &b, &c);
				if (a > b)
					swap(a, b);
				interval.change(a, b, 1 << (c - 1));
			}
			else if (op[0] == 'P')
			{
				scanf("%d %d", &a, &b);
				if (a > b)
					swap(a, b);
				int res = interval.query(a, b);
				printf("%d\n", bitcount(res));
			}
		}
	}
}

namespace p539 {

	const int MAXN = 100009;
	int n, m;
	int a[MAXN];

	struct Interval
	{
		int l, r;
		LL mx, he;
		Interval* lc, * rc;

		Interval(int l_, int r_)
		{
			l = l_; r = r_;
			mx = he = 0;
			lc = rc = NULL;
			if (l == r) {
				mx = he = a[l];
				return;
			}
			int mid = (l + r) / 2;
			lc = new Interval(l, mid);
			rc = new Interval(mid + 1, r);
			update();
		}

		void update()
		{
			mx = max(lc->mx, rc->mx);
			he = lc->he + rc->he;
		}

		LL query(int A, int B)
		{
			if (l > B || r < A) // 该线段在[A,B]外
				return 0;
			if (l >= A && r <= B) // 该线段在[A,B]中
				return he;
			return (lc->query(A, B) + rc->query(A, B));
		}

		void change(int A, int B)
		{
			if (l == r) {
				he = sqrt(he);
				mx = he;
				return;
			}
			int mid = (l + r) / 2;
			if (mid >= A && lc->mx > 1)
				lc->change(A, B);
			if (mid + 1 <= B && rc->mx > 1)
				rc->change(A, B);
			update();
		}

	};

	void read_case() {

	}
	void solve() {
		scanf("%d", &n);
		for (int i = 1; i <= n; i++)
		scanf("%d", &a[i]);
		Interval interval(1, n);
		scanf("%d", &m);
		for (int i = 1; i <= m; i++)
		{
			int op, a, b;
			scanf("%d", &op);
			if (op == 0)
			{
				scanf("%d %d", &a, &b);
				interval.change(a, b);
			}
			else if (op == 1)
			{
				scanf("%d %d", &a, &b);
				printf("%lld\n", interval.query(a, b));
			}
		}
	}
}

namespace p53A {

	const int MAXN = 50009;

	int n, m;
	int a[MAXN];

	void read_case() {

	}

	const int INF = 100000000;
	struct S1 {
		int he, lmax, rmax, f;
	};

	struct Interval
	{
		int l, r;
		int he,lmax,rmax,f; // max:最大子段和 lmax:以左端点为起点的最大和 rmax:以右端点为终点的最大和
		Interval* lc, * rc;

		Interval(int l_, int r_)
		{
			l = l_; r = r_;
			lc = rc = NULL;
			if (l == r) {
				he = lmax = rmax = f = a[l];
				return;
			}
			int mid = (l + r) / 2;
			lc = new Interval(l, mid);
			rc = new Interval(mid + 1, r);
			update();
		}

		S1 query(int A, int B)
		{
			if (l >= A && r <= B) {// 该线段在[A,B]中
				return { he,lmax,rmax,f };
			}
			S1 res1, res2;
			res1 = res2 = { 0,-INF,-INF,-INF };
			int mid = (l + r) / 2;
			if(mid >= A)
				res1 = lc->query(A, B);
			if(mid + 1 <= B)
				res2 = rc->query(A, B);
			S1 res;
			res.he = res1.he + res2.he;
			res.lmax = max(res1.lmax, res1.he + res2.lmax);
			res.rmax = max(res2.rmax, res2.he + res1.rmax);
			res.f = max(max(res1.f, res2.f), res1.rmax + res2.lmax);
			return res;
		}

		void update()
		{
			lmax = max(lc->lmax, lc->he + rc->lmax);
			rmax = max(rc->rmax, rc->he + lc->rmax);
			f = max(max(lc->f, rc->f), lc->rmax + rc->lmax);
			he = lc->he + rc->he;
		}
	};

	void solve() {
		scanf("%d", &n);
		for (int i = 1; i <= n; i++)
			scanf("%d", &a[i]);
		Interval interval(1, n);
		scanf("%d", &m);
		for (int i = 1; i <= m; i++)
		{
			int a, b;
			scanf("%d %d", &a, &b);
			S1 s = interval.query(a, b);
			printf("%d\n", s.f);
		}
	}
}

}

