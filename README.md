## 2023Fall_Projects

# Analyzing Oligopoly Competition in Game Theory using Monte Carlo Simulation
## The Cournot Model

### IS - 597 Final Project By: 
Ayush Majumdar 
ayush10@illinois.edu

## Abstract 
This project presents a comprehensive analysis of the Cournot
model, integrating concepts from statistics and micro-economics through
Monte Carlo simulation. The study employs a rigorous approach,
randomizing all variables involved and conducting simulations over
10,000 iterations. By blending these interdisciplinary perspectives, we
aim to unravel nuanced insights into the strategic interactions among
firms in oligopolistic markets. This research extends beyond traditional
theoretical frameworks, providing a dynamic and empirically grounded
exploration of market behaviors. The Monte Carlo simulation serves as a
powerful tool to capture the complexity and uncertainty inherent in
real-world scenarios, offering a robust foundation for decision-making
and policy formulation in the realm of industrial organization and
competition.

## Introduction 
The Cournot model\[1\], a cornerstone in industrial
organization and game theory, offers a fundamental framework for
analyzing strategic interactions among firms in oligopolistic markets.
Traditionally rooted in economic theory, this project expands the
analytical lens by incorporating concepts from statistics and game
theory. By employing Monte Carlo simulation, a versatile computational
tool, we introduce a dynamic and empirically grounded approach to
explore the intricacies of the Cournot model. Unlike deterministic
analyses, Monte Carlo simulation allows for the randomization of key
variables, reflecting the inherent uncertainty and complexity of
real-world economic systems.

The decision to conduct simulations over 10,000 iterations is driven by
the quest for robustness and precision in our findings. This extensive
iteration count ensures a thorough exploration of the parameter space,
yielding statistically reliable results. The randomized variables
encompass not only market demand and cost structures but also strategic
choices made by competing firms. This holistic approach captures the
multifaceted nature of strategic decision-making in oligopolistic
settings, where firms must consider not only their own actions but also
anticipate and react to the moves of their competitors.

This interdisciplinary approach, blending insights from game theory,
aims to transcend traditional limitations of the Cournot model. By
integrating statistical rigor and computational sophistication, our
analysis seeks to uncover novel perspectives on market behaviors and
strategic outcomes. The outcomes of this research hold implications for
industrial organization, competition policy, and decision-making in
complex economic environments. Through this innovative fusion of
methodologies, we contribute to a richer understanding of strategic
interactions in oligopolistic markets, paving the way for more informed
policy decisions and strategic business planning.

## Null Hypothesis 
1. There is no significant difference in the average profits of Firm 1 and Firm 2 under Cournot competition. In other words, the profits of both firms are equal in the equilibrium.
2. There are no significant differences in the working of the Cournot model under elastics as well as inelastic demands.
3. Under real world scenarios which include random real world factors such as demand and supply shock the profits gained by the firms will follow a normal distribution adhering to the Central Limit Theorem.

## Methodology 
The methodology for this project involves employing Monte
Carlo simulation to analyze the Cournot model, incorporating concepts
from statistics, economics, and game theory. The simulations aim to
investigate the strategic interactions among firms in oligopolistic
markets by randomizing key variables. The following template outlines
the essential components of each simulation. 
\* Variables: 
1. Number of iterations
2. Demand curve coefficient
3. Cost of production (Ci)
4. Quantity of Production
5. Market Demand

## Novelty 
1. Studying the Cournot Model in an Oligopolic Duopoly from an analytical point of view instead
of a mathematical point of view.
2. Use of Monte Carlo Simulation to simulate the Oligopoly Market for a large duration.
3. Simplify the Cournot model profit maximization using simple economic demand and supply equations.
4. Modeling the Nash Equilibrium\[2\]\[3\] into a Monte Carlo Simulation to ensure both the firms profit under all 10000 iterations of the simulation.
5. The Cournot Model has previously been studied under isoelastic conditions\[6\]. However, this study examines its effects under both elastic and inelastic demands.
6. Modeling the Cournot Model under 6 different real world scenarios:
      \* Demand Shock \*Supply Shock \* Inventory \* Disease Outbreak (p=0.1) \* Inflation(p=0.7)

## ChatGPT Queries 
### Q: explain elastic demand briefly 
A: Elastic demand refers to a situation in which the quantity demanded of a good or
service is highly responsive to changes in price. In other words, when
the price of a product changes, the percentage change in the quantity
demanded is relatively larger. Elasticity of demand is a measure of how
sensitive the quantity demanded is to a change in price.

Key characteristics of elastic demand include:

1\. \*\*Elasticity Coefficient Greater Than 1:\*\* The price elasticity
of demand (PED) is calculated as the percentage change in quantity
demanded divided by the percentage change in price. In elastic demand,
the elasticity coefficient is greater than 1, indicating a relatively
larger response in quantity demanded to a change in price.

\\\[ \\text{PED} = \\frac{\\%\\ \\text{Change in Quantity
Demanded}}{\\%\\ \\text{Change in Price}} \\\]

2\. \*\*Substitute Availability:\*\* Elasticity often depends on the
availability of substitutes. If there are readily available substitutes
for a product, consumers are more likely to switch to alternatives if
the price of the original product changes.

3\. \*\*Luxury Goods:\*\* Goods and services that are considered
luxuries often exhibit elastic demand. Consumers are more responsive to
price changes for luxury items because they can easily forego or
substitute them with other options.

4\. \*\*Longer Time Horizons:\*\* Elasticity can vary over different
time horizons. In the long run, consumers may have more options and
flexibility to adjust their consumption patterns in response to price
changes, leading to more elastic demand.

Graphically, elastic demand is represented by a relatively flat demand
curve. When the price changes, the quantity demanded moves
proportionately in the opposite direction. The midpoint formula is often
used to calculate percentage changes in quantity and price to determine
elasticity.

Understanding elasticity is crucial for businesses and policymakers to
predict how changes in price will impact total revenue and consumer
behavior. If demand is elastic, a price increase may lead to a
significant decrease in total revenue, while a price decrease may result
in a substantial increase in total revenue. Explanation: Query to
understand elastic demand

### Q: How would the cournot model change with elastic and inelastic demand?
A: The Cournot model is a simple economic model used to analyze
competition between firms in an oligopoly market. In the Cournot model,
firms simultaneously choose the quantity of output they will produce,
taking into account the output levels chosen by other firms in the
market. The model assumes that firms are price setters rather than price
takers.

The demand elasticity plays a crucial role in determining the impact of
changes in market conditions on the equilibrium outcomes in the Cournot
model. Let\'s consider the effects of both elastic and inelastic demand
on the Cournot model:

1\. \*\*Elastic Demand:\*\*  - In an elastic demand scenario, consumers
are relatively responsive to changes in price. This means that if a firm
increases its price, the quantity demanded will decrease significantly,
and vice versa.  - With elastic demand, firms in the Cournot model will
have a more pronounced incentive to reduce their output levels. This is
because a reduction in output will lead to a smaller increase in price,
resulting in a relatively smaller decrease in quantity demanded.  - The
equilibrium quantity will be lower, and prices will be closer to the
competitive outcome compared to a situation with inelastic demand.

2\. \*\*Inelastic Demand:\*\*  - In contrast, with inelastic demand,
consumers are less responsive to changes in price. This implies that
changes in price will have a smaller impact on the quantity demanded.  -
In the Cournot model with inelastic demand, firms will have less
incentive to adjust their output levels in response to changes in market
conditions. The quantity produced by each firm may be higher, and prices
may deviate more from the competitive outcome.  - The equilibrium
quantity will be higher, and prices will be farther from the competitive
outcome compared to a situation with elastic demand.

In summary, the elasticity of demand influences the strategic
interactions among firms in the Cournot model. Elastic demand tends to
lead to lower output levels and prices closer to the competitive
outcome, while inelastic demand tends to result in higher output levels
and prices farther from the competitive outcome. The specific outcomes
will depend on the elasticity of demand and the number of firms in the
market. Explanation: This query helped me validate the simulation
results and also helped me select values for the demand and supply
curve\[9\] as well as the elasticity constant.

### Q: Is alpha value high for inelastic demand? 
A: In the context of the Cournot model, the parameter α (alpha) represents the intercept of the
demand curve. A higher value of α implies a higher level of demand when
the quantity supplied is zero. The specific value of α alone doesn\'t
inherently indicate whether the demand is elastic or inelastic; it is
the combination of α and the slope parameter β that determines the
elasticity of demand. Inelastic demand typically refers to a situation
where consumers are less responsive to changes in price. This is
reflected in a smaller absolute value of the slope parameter (\|β\|) in
the demand curve equation. In the Cournot model, a lower \|β\| tends to
lead to inelastic demand. \* If α is high and \|β\| is relatively low,
you have a situation where demand is less responsive to changes in
quantity (QQ), and this could be interpreted as inelastic demand. \* If
α is high and \|β\| is relatively high, you have a situation where
demand is more responsive to changes in quantity, and this could be
interpreted as elastic demand. It\'s important to note that the
relationship between α, β, and elasticity is not linear. Both parameters
contribute to determining the elasticity of demand, and you need to
consider their combined effect. Additionally, the specific
interpretation may depend on the units used for quantity and price.
Explanation: This query helped me understand how the alpha and beta
values worked with each other.

## Citations 
1. G. Romeo, 'Chapter 6 - Microeconomic theory in a static
environment', in Elements of Numerical Mathematical Economics with
Excel, G. Romeo, Ed. Academic Press, 2020, pp. 295--382.
2. https://eml.berkeley.edu/\~webfac/card/e101a_s05/101duopolytheory.pdf
3. http://cramton.umd.edu/econ414/ch03.pdf
4. https://www0.gsb.columbia.edu/faculty/nsicherman/B7006-002/duopoly.pdf
5. https://www.eco.uc3m.es/\~mmachado/Teaching/OI-I-MEI/slides/3.2.Cournot%20Model.pdf
6. L. Fanti, L. Gori, and M. Sodini, 'Nonlinear dynamics in a Cournot
duopoly with isoelastic demand', Mathematics and Computers in
Simulation, vol. 108, pp. 129--143, 2015.
7. https://www.studysmarter.co.uk/explanations/microeconomics/imperfect-competition/cournot-model/
8. https://www.investopedia.com/terms/c/cournot-competition.asp
9. https://www.investopedia.com/terms/l/law-of-supply-demand.asp
