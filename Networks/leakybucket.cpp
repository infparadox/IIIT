#include<bits/stdc++.h>
using namespace std;
typedef long long int ll;

int main()
{
	ios::sync_with_stdio(0);
	ll q;
	cin >> q;
	while(q--)
	{
		ll k,c,n,t,i,j,sum=0,num_drop=0;
		cin >> k >> c >> n >> t;
		ll a[k+1];
		for(i=1;i<=k;i++)
			cin >> a[i];
		deque<ll>q,dropped;
		i=1;
		while(i<=k || !q.empty())
		{
			if(i<=k && sum+a[i]<=c)
			{
				q.push_back(a[i]);
				sum+=a[i];
			}
			else if(i<=k && sum+a[i]>c)
			{
				dropped.push_back(a[i]);
				num_drop++;
			}
			if(i%t==0)
			{
				ll tem=0;
				while(tem<=n)
				{
					if(q.empty())
						break;
					tem+=q.front();
					if(tem<=n)
					{
						ll x=q.front();
						cout << x << " " << i << "\n";
						sum-=x;
						q.pop_front();
					}
				}
			}
			i++;
		}
		if(num_drop==0)
			cout << "0";
		while(!dropped.empty())
		{
			cout << dropped.front() << " ";
			dropped.pop_front();
		}
		cout << "\n";
	}
	return 0;
}