# FiniteTemperatureLanczos


```math
\langle A \rangle = \sum_{n=1}^{N_{st}} e^{-\beta E_n} \langle \Psi_{n} \vert A \vert \Psi_{n} \rangle \Big/ \sum_{n=1}^{N_{st}} e^{-\beta E_n}
```

For a general orthonormal basis $\vert n \rangle$,
```math
\langle A \rangle = \sum_{n=1}^{N_{st}} \langle n \vert e^{-\beta H} A \vert n \rangle \Big/ \sum_{n=1}^{N_{st}} \langle n \vert e^{-beta H} \vert n \rangle
```

High temperature expansion

```math
\begin{aligned}
\langle A \rangle &= Z^{-1} \sum_{n=1}^{N_{st}} \sum_{k=0}^{\infty} \frac{ (-\beta)^k }{k!} \langle n \vert H^k A \vert n \rangle \\
Z &= \sum_{n=1}^{N_{st}} \sum_{k=0}^{\infty} \frac{ (-\beta)^k }{k!} \langle n \vert H^k \vert n \rangle 
\end{aligned}
```

Terms in the expansion $\langle n \vert H^{k} A \vert n \rangle$ can be calculated exactly using the Lanczos procedure with $M \ge k$ steps.
We get
```math
\langle n \vert H^{k} A \vert n \rangle = \sum_{i=0}^{M} \langle n \vert \psi_{i}^{n} \rangle \langle \psi_{i}^{n} \vert A \vert n \rangle (\epsilon_{i}^{n})^{k}
```
where $\psi_{i}^{n}$ is the $i$th basis state of the Krylov space of $\vert n \rangle$.
