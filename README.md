Demo of personalization on Tinfoils private enclaves

To get the most out of personalization, you need your data to be private. For an example personalization task we use Prime Intellect's prime-RL framework, and a community made environment `apaz/writing-style-matching`.

This server will expose an endpoint to train a model, and then to do inference with your trained model. Lora weights will live on encrypted memory, unless it gets too be too much in which case we'll encrypt them and store them to disk.

As a user you'll authenticate with an encryption key, and use the demo key we give you. YOu'll send both of these on every request.

TBD on how we'll do the SDK or if we'll have a FE or what.

## Architecture
