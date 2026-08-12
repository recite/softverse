# Draft: report to Harvard Dataverse support

Not sent. For Gaurav to review, edit and send to <support@dataverse.harvard.edu>
(or to file on <https://github.com/IQSS/dataverse.harvard.edu/issues>, where
issue #468 looks like the same underlying cause).

---

**Subject:** API requests receive AWS WAF challenge (HTTP 202, empty body) — authenticated token never evaluated

Hello,

I am doing research on software use in social science replication code, using
the Harvard Dataverse API to read metadata and files from journal collections. I
have an API token and it authenticates correctly (`/api/users/:me` returned my
account earlier today).

Since a few hours ago, every request from my client receives:

```
HTTP/2 202
server: awselb/2.0
content-length: 0
content-type: text/html; charset=UTF-8
x-amzn-waf-action: challenge
access-control-expose-headers: x-amzn-waf-action
```

An AWS WAF challenge, with an empty body. Details that may help narrow it down:

- **It is not endpoint-specific.** `/api/info/version`, `/api/search`,
  `/api/datasets/:persistentId/versions/:latest-published`,
  `/api/access/datafile/{id}`, the `/dataset.xhtml` page and even `/robots.txt`
  all return the same 202 challenge.
- **`/oai` is the exception** — `https://dataverse.harvard.edu/oai?verb=Identify`
  returns 200 normally. So OAI-PMH appears to be allowlisted while everything
  else is challenged.
- **It is not specific to my client or my network.** `curl` (with and without a
  User-Agent), Python `httpx`, and a fetch from an unrelated network all get the
  same response.
- **Authentication makes no difference.** Anonymous and token-authenticated
  requests behave identically, which suggests the WAF rule runs before the API
  token is evaluated.
- Requests from the same client succeeded normally earlier the same day, so it
  looks like a threshold is being crossed rather than a permanent rule.

Two things I would appreciate guidance on:

1. **Is this intended for API clients?** A 202 with an empty body is
   indistinguishable from success to any standard HTTP library, so a
   well-behaved client will record an empty result and treat it as complete
   rather than retry. If the intent is to rate-limit, a 429 with `Retry-After`
   would let clients back off correctly. I am happy to file this as a GitHub
   issue if that is more useful — it may be the same cause as
   IQSS/dataverse.harvard.edu#468.

2. **What is the sanctioned way to do this kind of bulk read?** I need version
   metadata and code files for roughly 13,800 datasets across the journal
   collections. My client already limits itself to ~2 requests/second with
   evenly spaced requests, uses `/api/access/datafiles/{ids}` to fetch a whole
   dataset's scripts in a single request rather than one per file, retries only
   429/5xx, and never retries 404 or 403. I am glad to slow down further, run
   at off-peak hours, or work to whatever limit you would prefer. If there is an
   allowlisting process or a preferred bulk route, I will use it.

The resulting dataset and code will be released publicly, and I would be glad to
acknowledge Harvard Dataverse appropriately.

Thank you,

Gaurav Sood
<gsood07@gmail.com> · Dataverse account: `@soodoku`
