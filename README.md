# Doctown v6

## QUICK INFO

Here are some quick bits of info pertinent to the whole project:

- Doctown Docker repository: `xandwrp/doctown:latest`
- RunPod Instance ID: [unset, need to make Dockerfile to deploy to start up a runpod instance first]

---

## What I've learned
Okay, so v6 now... this is moving fast. And I've learned a lot.
The key thing being this: I can deploy my ENTIRE pipeline to just one GPU-based serverless RunPod instance.
That's it. No need for a separate CPU only builder pipeline. Because the GPU pods ALREADY HAVE CPUs in them...

So here's the plan.

## The Builder
This is the Docker image that will be deployed and run on the GPU pod. It will handle the ENTIRE ingest and process pipeline.
It should be fed in one thing--a GitHub repo. Just public for now to avoid OAuth headaches.
And it should ingest the repo, download it locally, unzip it, and for now just return a JSON containing the file structure
of the ingested repo to prove we successfully unzipped and "processed" it to the user.

This can be tested end to end with just one serverless RunPod GPU instance and one Dockerfile.
No website, no frontend, we can debug with direct CURL or scripts on the endpoint.

Technically I don't even NEED RunPod for this. Just a central local pipeline I can type a git URL into and have spit out the 
file structure automatically. This is what will be deployed and interacted with on RunPod.

I think I'll call the Docker image just... Doctown. xandwrp/doctown:latest. Because this one image WILL BE ALL OF Doctown's pipeline
condensed into one neat deployment strategy. And I really like that simplicity. The real architecture of this design only revealed itself
after many "entanglements" with multple deployments and the headache they caused me. This is simple, and simple is really fucking good. 

## The Website
This is a secondary feature to let others use the platform, but I won't worry about it at all until I can verify the builder
pipeline is working correctly. This is a phase 2 development.