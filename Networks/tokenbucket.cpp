#include<bits/stdc++.h>
using namespace std;
typedef long long int ll;

int main()
{
	ll test;
	cin >> test;
	for(ll z=0;z<test;z++)
	{
		double in_rate,in_size,buc_cap,t_rate,out_rate;
		double total_time=0;
		scanf("%lf %lf %lf %lf %lf",&in_rate,&in_size,&buc_cap,&t_rate,&out_rate);
		if(in_rate > t_rate)
		{
			if((t_rate < in_rate) && (in_rate < out_rate))
			{
				double t1 = (buc_cap)/(in_rate - t_rate);
				double tem = t1*in_rate;
				if(tem<=in_size)
				{
					total_time = ((in_size - (t1*in_rate))/(t_rate));
					total_time+=t1;
				}
				else
					total_time = in_size/(in_rate-t_rate);
			}
			else
			{
				double t1 = (buc_cap)/(out_rate-t_rate);
				double tem = t1*out_rate;
				if(tem<=in_size)
				{
					total_time = ((in_size - (t1*out_rate))/(t_rate));
					total_time+=t1;
				}
				else
					total_time = in_size/(out_rate-t_rate);
			}
		}
		else if(in_rate<=t_rate)
			total_time = (in_size/in_rate);
		total_time=total_time*1000;
		cout << total_time << "\n";
	}
	return 0;
}