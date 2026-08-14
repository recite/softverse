# Harvard Dataverse: access note and draft email

## What we found

Harvard Dataverse returns `HTTP 202` with an empty body and
`x-amzn-waf-action: challenge` to every request from a plain HTTP client, on
every route including the homepage. It is a JavaScript capability test: the
client is asked to run a script and retry. There is no CAPTCHA and no human
step.

The mitigation sits at the load balancer, ahead of the application, so it
never sees a credential. Measured, all byte-identical:

| request | result |
|---|---|
| `/api/info/version`, anonymous | 202, empty |
| `/api/info/version`, `X-Dataverse-key` header | 202, empty |
| `/api/info/version`, `?key=` query parameter | 202, empty |
| `/api/users/:me` with a valid token, curl | 202, empty |
| `/api/users/:me` with the same token, browser | `{"status":"OK","identifier":"@soodoku"}` |

That is why a working API token appeared not to work.

`/oai` is *not* behind it, but cannot substitute: its 42 sets are
institutional (`AfricaRice`, `boston_college`, `CIFOR`) and no journal
dataverse is exposed, so the frame is only reachable through `/api`.

## What the terms actually say

Quoted from the [Harvard Dataverse API Terms of
Use](http://best-practices.dataverse.org/harvard-policies/harvard-api-tou.html).
In using the APIs, "you shall not":

- "attempt to conceal or otherwise misrepresent your identity or your
  application's identity when using or requesting authorization to use the
  Dataverse APIs"
- "use an unreasonable amount of bandwidth. Harvard Dataverse reserves the
  right to decide what is unreasonable"
- "use Dataverse APIs in a manner that may impair the functionality,
  stability, or operation of Harvard Dataverse servers or adversely impact
  the behavior of other users"

Automated access is not prohibited. Neither is executing a challenge. The
binding constraints are identity, bandwidth and impairment — which is why the
client sends our own user-agent rather than a browser's default (the first
clause makes that mandatory, not decorative), authenticates so requests are
attributable, and rate-limits to one request per two seconds.

An earlier version of `softverse/acquire/http.py` asserted that satisfying
the challenge "would be circumventing an access control" and cited these
terms for it. That was an inference presented as their position, and it is
corrected.

## Status

The frame collected cleanly: **12,336 deposits across 74 collections**, one
request per collection, about seventy-five requests in total. Counts run
consistently above the 2024 scrape — `ajps` 836 against 677, `the_review` 732
against 540, `xps` 239 against 178 — which is the growth two years should
produce and shows no systematic shortfall.

One collection, `restat`, returned nothing on the first pass, and the draft
email below originally told Harvard it might be a bug on their side. It was
not. The alias is live, its DOIs resolve, and the cause was our own timeout:
the run averaged 23.3 s per collection against a 25 s budget, so every
collection sat at the ceiling and the largest one fell off it. `jop` cleared
it at 1,219 deposits; `restat` has roughly 1,600.

Two things came out of that worth keeping. The budget is now set from the
measurement rather than a guess, with one retry at four times the budget
before anything is called a failure. And the client no longer reports an
empty response as a *challenge*: `--dump-dom` returns no headers, so a
challenge, a timeout, a redirect and a dead network are indistinguishable to
it, and asserting one of them is precisely how a bug of ours nearly became an
accusation in a letter.

## Draft email

> **To:** support@dataverse.harvard.edu
> **Subject:** Research use of the Dataverse API — asking about rate and access
>
> Hello,
>
> I'm Gaurav Sood. With Daniel Weitzel I'm doing a study of which software
> research actually runs on, by parsing the code in replication deposits
> across R, Python and Stata. Harvard Dataverse is the main home for social
> science replication material, so it matters a great deal to the work.
>
> I wanted to tell you what we're doing and ask whether you'd prefer we did
> it differently.
>
> We need two things from the API. First, the dataset listing for each
> journal dataverse — `/api/dataverses/{alias}/contents`, roughly seventy-five
> requests in total, one per journal, which we have now run. Second, and only
> if you're comfortable with it, the file listings and public file downloads
> for those deposits, which is the larger ask.
>
> Requests are authenticated with my API token, carry the user-agent
> `softverse/2.0 (research; github.com/recite/softverse)` so they are
> identifiable in your logs, and are rate-limited to one every two seconds.
> We stop on any 429 and would stop immediately if you asked.
>
> One practical note: your bot-mitigation layer returns HTTP 202 with an
> empty body to any client that does not execute JavaScript, on every route,
> and it sits ahead of authentication so an API token makes no difference.
> That makes the documented API unusable from ordinary HTTP clients. If
> there's an allowlist for research use, or a bulk metadata export you'd
> rather we used, we'd much prefer either to working around it.
>
> Happy to describe the project in more detail, share what we build, or
> schedule the work for whenever suits your infrastructure.
>
> Thanks,
> Gaurav Sood — gsood07@gmail.com — Dataverse account @soodoku
