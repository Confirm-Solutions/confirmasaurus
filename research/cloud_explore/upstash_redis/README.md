## Why Upstash Redis?

**Update 12/12/2022**: We are not currently using Upstash or Redis, but this is still useful info**

Clickhouse can't satisfy all our coordination needs because it doesn't have any mechanism for locking. So we might use a very lightweight Redis server through Upstash in addition to Clickhouse. This is almost entirely for distributed locks at the moment. But, it would be entirely reasonable to use this Redis instance for other data that we want low latency access to. Why Upstash/Redis for a distributed lock server? Because I can't find another service that is 1) as easy to set up and 2) free. 3) very low latency. Here is a link to some [database latency benchmarks](https://serverless-battleground.vercel.app). [And a discussion of some pros and cons of Upstash's competitors.](https://upstash.com/blog/best-database-for-serverless)

The distributed locking implementation we are using is called Redlock (Redis Lock) and the implementation is through the pottery package (https://github.com/brainix/pottery#redlock). There is [some mild controversy](https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html) about whether Redlock is a good implementation of distributed locking. I tend to agree with Martin Kleppmann that Redlock is insufficient for use cases where correctness depends on the lock working perfectly 100% of the time. But, we are not in that situation. He specifically says that for our situation (lock is just saving computational effort), that Redlock is entirely sufficient. 

Clickhouse vs Redis/Upstash:
- Clickhouse is for large data. Also, it performs just fine for small data where we don't need strong consistency. 
- Redis is for low latency small data. Currently, only distributed locks. But you can use it for other stuff too if needed.
