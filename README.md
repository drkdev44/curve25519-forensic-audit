<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/02cdedfb-64ba-49a3-a113-c8ff08be896e" />
All models scored around 50%, which means we found no operating system or architecture fingerprints.

Q.What does this experiment tries to achieve?
A. By generating 1000 keys from Windows ARM and Linux x86 we try to test if the keys would be seperable because of ISA & OS differences.
----------------------------------------------------------------------------------------------------------------------------------------
Current Important Aim : Porting the project to faster languages like C/Rust and adding much more features. Then choosing and re-implementing the ML code so it would make better inferences. Also roughly 100 Million to several Billions of public keys need to be generated for each both linux-x86 and windows-arm to see if very small but calculable differences exists. Even my spesific choice for amplifying the operating system and architecture impact of rng might be not enough for analyzing.
