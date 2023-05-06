# tiny-slack-gpt

Single file app to run a GPT slackbot
Includes basic Text2SQL capabilities

Use manifest.json or follow the following steps.

- create application on slack
- enable socket mode
- create OAuth tokens
- add slash commands for /query and /summarize
- figure out your bot user ID
- enable the right scopes:

```
Bot Token Scopes    |    User Token Scopes
=============================================      
app_mentions:read   |    channels:history
channels:history    |    chat:write
channels:read       |    groups:history
chat:write          |    im:history
commands            |    im:read
groups:history      |    im:write
groups:read         |    mpim:history
im:history          |    mpim:read
mpim:history        |    mpim:write
users:read          |
```

- run from terminal, cron, or as a service
