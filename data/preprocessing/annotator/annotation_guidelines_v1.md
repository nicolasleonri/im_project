title: "Annotation Guidelines for Claims and Premises",
abstract: [These guidelines specify how to identify and annotate claims and premises in argumentative texts. They are intended for manual corpus annotation and for informing computational argument mining systems. The focus is limited to identifying argumentative components, not their relations.],

= Scope and Assumptions
The corpus consists of sentences extracted from Peruvian Spanish-language newspaper articles in the domain of the informal economy. The informal economy comprises economic activities that operate outside formal labor, tax and regulatory frameworks but involve legal goods and services—distinct from illegal or intrinsically criminal activities. This includes self-employed workers, wage earners in non-compliant firms, and enterprises lacking legal registration @williams_lansky_2013 @chen_2012.

Each article includes *title, content, and metadata* (e.g., date, outlet, author, section) and is associated with a unique `article_id`. Both title and content are further segmented at the sentence level, with each sentence receiving a unique `sentence_id`. Annotation is performed at the Elementary Discourse Unit (EDU) level, corresponding to individual sentences or titles. Both title and content may contain explicit and/or implicit argumentative material.

Annotators are allowed to perform minimal reconstruction of implicit claims or premises when the inferential link is strongly supported by the immediate textual context. However, reconstruction must remain local and textually grounded. No global reconstruction of the author’s overall thesis or speculative interpretation beyond the surrounding discourse is permitted. If the argumentative function cannot be reasonably inferred from the local context, the segment should be labeled as `none`.

== Domain Definition: Informal Economy <sec_informal_economy>
For the purposes of this annotation task, the concept of informal economy is grounded in the theoretical tradition initiated by #cite(<Hart19708SE>, form: "author") (#cite(<Hart19708SE>, form: "year")), who introduced the term to describe *economic activities that take place outside the framework of official institutions*. From a conceptual perspective, the informal economy includes activities that are not regular, predictable, reproducible or institutionally recognized, in contrast to _formal_ economic practices regulated by state institutions, capitalist organizations and dominant economic discourse.

Subsequent literature has emphasized the inherent heterogeneity and analytical openness of the concept. As noted by #cite(<hart_1985>, form: "author") (#cite(<hart_1985>, form: "year")), informality does not exist as a directly observable empirical object, but rather as a relational category defined in contrast to what is considered the orthodox or formal core of the economy. Accordingly, concepts of informality vary depending on whether the analytical focus lies on enterprises, employment relationships or specific economic activities @williams_lansky_2013.

In line with the current consensus reflected in the International Labour Organization (ILO) framework, the informal economy is understood to comprise:
+ Self-employed workers operating in informal enterprises;
+ Employers working in their own informal enterprises;
+ Contributing family workers in both formal and informal sectors;
+ Wage earners holding informal jobs in formal enterprises, informal enterprises, or households;
+ Own-account workers engaged in the production of goods exclusively for household consumption.

This definition explicitly distinguishes informal economic activities from illegal or criminal economies. Informality concerns the production or commercialization of legal goods and services conducted outside formal regulatory, labor or tax frameworks, rather than intrinsically criminal activities. The literature also proposes multiple theoretical perspectives to explain the emergence and persistence of informality —including modernizing (dualist), structuralist, neoliberal (legalist), postmodern, and voluntarist approaches— each highlighting different causal mechanisms and actor motivations. Rather than privileging a single explanatory framework, annotation should remain neutral with respect to these perspectives and reflect how such positions are articulated, evaluated or contested within the journalistic text itself. 

Annotators should rely on this domain definition to disambiguate references to informality in journalistic discourse; interpret implicit claims and premises related to causes, consequences, or evaluations of informal economic practices and determine whether a segment falls within the thematic scope of the corpus.

= Definitions
== Claims
A claim is a proposition expressing a standpoint that the author (or a quoted actor within the text) advances and intends the reader to consider, accept or evaluate. Claims represent positions that are potentially contestable and function as central points within an argumentative structure @stede2019argumentation.

Annotate a sentence as a *claim* if it:
- States an opinion, evaluation, judgment, recommendation or conclusion.
- Represents a position that could reasonably be questioned or opposed.
- Functions as the point that other segments justify, support or criticize.

Typical realizations of claims include:
- Evaluative statements: "_Esta política es ineficaz._"
- Normative statements or calls for action: _"El edificio debería ser demolido."_
- Predictive or diagnostic judgments: _"Esto causará daños a largo plazo."_
- Explicit conclusions, often signaled by markers: _por lo tanto_, _también_ or _por ende_.

Each sentence is labeled individually; relationships between claims are not annotated at this stage. Claims may be expressed in the journalist’s voice or through reported speech (e.g., experts, unions, authorities). When the reported content itself expresses an evaluative, normative or diagnostic standpoint, annotate the sentence as `claim`, regardless of whether the journalist explicitly endorses it.

== Premises
A premise is a proposition that provides justificatory support for a claim by supplying evidence, causal explanation, authority reference, statistical data, or illustrative example. Premises answer an implicit or explicit "Why?" question relative to another segment.

Annotate a sentence as a *premise* if it:
- Explains why a claim should be accepted
- Provides a reason why a claim should be accepted.
- Supplies evidence, data, examples or causal explanations.
- Answers an implicit or explicit _"Why?"_ question with respect to another segment.

Typical realizations of premises include:
- Factual assertions: _"El edificio contiene asbesto."_
- Causal statements: _"Porque supone un riesgo para la salud"_
- Appeals to authority: _"Según el alcalde/especialista..."_.
- Illustrative: supporting examples for a generalization

Premises may support other premises, forming sub-arguments. If no claim is explicitly present, label premises only when they clearly imply a claim, otherwise mark as `none`. In journalistic discourse, not all factual statements are premises. A factual or descriptive sentence should be annotated as premise only if it functions as a reason supporting a claim within the local discourse context. Pure background information or contextual description without inferable argumentative function should be labeled as `none`.

= Annotation Instructions

Based on #cite(<peldzus_stede13>, form: "author") (#cite(<peldzus_stede13>, form: "year")), the workflow is:
+ *Read EDUs*: smallest meaningful text units (sentences or clauses)
+ *Determine ADUs*: check if an EDU is argumentative
  - If not, label as `none`
+ *Label ADUs*: classify as `claim` or `premise`

When distinguishing between labels, apply:
- #underline[Support Test]
  - If the segment is primarily supported by other segments, annotate it as a `claim`.
  - If the segment primarily supports another segment, annotate it as a `premise`.
- #underline[Why-Test]
  - If the segment can naturally answer the question “Why?” with respect to another segment, it is a `premise`.
- #underline[Challenge Test]
  - If the segment expresses a standpoint that the author expects disagreement with, annotate it as a `claim`.
- #underline[Conclusion Marker Test]
  - Segments introduced or framed as conclusions are annotated as `claim`.

== Special Cases
- *Multiple Claims*: Annotate each distinct standpoint as a Claim, even if multiple claims occur within the same paragraph or sentence.
- *Restated Claims:* If a claim is repeated or paraphrased, annotate each occurrence as a Claim. Equivalence can be marked separately if supported by the annotation format.
- *Rhetorical Questions:* Rhetorical questions that clearly imply a standpoint should be annotated as Claims, using their implied propositional content.
- *Non-Argumentative Segments:* Do not annotate background info, organizational text, or narrative without argumentative function
- *Reported Expert Statements:* When a journalist quotes an expert or authority to advance an evaluative or explanatory position, annotate the sentence as `claim` if the quoted content itself expresses a standpoint. The strategic use of expert voice is considered an argumentative move at the sentence level.

=== Titles

Titles are annotated following the same functional criteria as body sentences.

- If the title expresses an evaluative, normative, contrastive or diagnostic standpoint, annotate it as `claim`.
- If the title merely reports an event or factual situation without evaluative framing, annotate it as `none`.
- Headlines that attribute a position to an actor (e.g., "Autoridades piden orden") are annotated as `claim` when they frame a standpoint within the article’s argumentative structure.
- When in doubt, apply the challenge test: if the title expresses a position that could reasonably be questioned or opposed, annotate it as `claim`.

= Examples
== Example 1
- #underline[Title:] _“Ambulantes invaden centro histórico: autoridades piden orden”_
- #underline[Content:] _“Los comerciantes informales han tomado las principales calles del centro de Lima, generando caos vehicular y afectando el comercio establecido. Los empresarios formales denuncian competencia desleal, pues los ambulantes no pagan impuestos ni cumplen normas sanitarias. ’Nosotros pagamos licencias, impuestos y planilla, mientras ellos venden lo mismo sin ningún costo’, señala un comerciante formal. Las autoridades municipales anunciaron operativos de fiscalización más estrictos para recuperar el espacio público.”_

#table(
  columns: 3,
  align: center + horizon,
  stroke: black,
[Sentence],
[Annotation],
[Justification],
[*Title:* “Ambulantes invaden centro histórico: autoridades piden orden”],
[none],
[Expresses the authorities' standpoint that order is needed, but not the author's standpoint.],
[“Los comerciantes informales han tomado las principales calles del centro de Lima, generando caos vehicular y afectando el comercio establecido.”],
[Premise],
[Evidence of the problem, explains why order is needed.],
[“Los empresarios formales denuncian competencia desleal, pues los ambulantes no pagan impuestos ni cumplen normas sanitarias.”],
[Claim],
[Standpoint: informal vendors create unfair competition.],
[“’Nosotros pagamos licencias, impuestos y planilla, mientras ellos venden lo mismo sin ningún costo’.”],
[Premise],
[Supports the claim with factual data about costs.],
[“Las autoridades municipales anunciaron operativos de fiscalización más estrictos para recuperar el espacio público.”],
[Claim],
[Expresses a recommendation for action (stricter controls).],
)

== Example 2
- #underline[Title:] “El emprendimiento informal: motor económico que necesita apoyo”
- #underline[Content:] “María vende jugos naturales en un carrito en San Juan de Lurigancho. Como millones de peruanos, enfrenta diariamente la burocracia que impide su formalización. ’Quise sacar mi RUC pero los trámites son eternos y los costos muy altos’, explica. Hernando de Soto ya lo advertía: el problema no son los informales sino las trabas del Estado. Estos emprendedores generan el 70 % del empleo nacional, pero carecen de acceso a créditos por no tener títulos de propiedad. Una simplificación administrativa podría liberar este enorme potencial económico y convertir a miles de Marías en empresarias formales que contribuyan al desarrollo del país.”

#table(
  columns: 3,
  align: center + horizon,
  stroke: black,
[Sentence],
[Annotation],
[Justification],
[*Title:* “El emprendimiento informal: motor económico que necesita apoyo”],
[Claim],
[Expresses the author’s viewpoint that informal entrepreneurship requires support.],
[“María vende jugos naturales en un carrito en San Juan de Lurigancho.”],
[Premise],
[Provides context for the claim (example of informal entrepreneur).],
[“Como millones de peruanos, enfrenta diariamente la burocracia que impide su formalización.”],
[Premise],
[Explains the barrier preventing formalization.],
[“'Quise sacar mi RUC pero los trámites son eternos y los costos muy altos', explica.”],
[Premise],
[Illustrative evidence of bureaucratic obstacles.],
[“Hernando de Soto ya lo advertía: el problema no son los informales sino las trabas del Estado.”],
[Claim],
[Standpoint: structural problem is with the state, not individuals.],
[“Estos emprendedores generan el 70 \% del empleo nacional, pero carecen de acceso a cr\'editos por no tener títulos de propiedad.”],
[Premise],
[Evidence supporting claim about structural barriers.],
[“Una simplificación administrativa podría liberar este enorme potencial económico y convertir a miles de Marías en empresarias formales que contribuyan al desarrollo del país.”],
[Claim],
[Prescriptive conclusion: administrative reform would unlock potential.],
)

== Example 3
- #underline[Title:] “Mercados comunitarios andinos: economía de reciprocidad frente al capitalismo”
- #underline[Content:] “En los mercados de Huancayo persiste una forma de economía que desafía la lógica capitalista. El ’ayni’ —sistema de reciprocidad ancestral— organiza el intercambio entre vendedores y compradores. Doña Juana no solo vende papas; participa en una red de solidaridad donde el dinero es solo una forma de intercambio entre muchas otras. ’Aquí nos ayudamos entre todos. Si alguien necesita, prestamos sin interés’, explica. Estos espacios informales no buscan maximizar ganancias sino garantizar la subsistencia colectiva. El antropólogo Carlos Montes señala: ’No es informalidad por falta de oportunidades, es una elección cultural que mantiene vivas tradiciones milenarias de cooperación’. Frente a la crisis del modelo neoliberal, estas economías alternativas ofrecen lecciones de sostenibilidad.”


#table(
  columns: 3,
  align: center + horizon,
  stroke: black,
[Sentence],
[Annotation],
[Justification],
[*Title:* “Mercados comunitarios andinos: econom\'\ia de reciprocidad frente al capitalismo”],
[Claim],
[Expresses the standpoint that these markets are alternatives to capitalism.],
[“En los mercados de Huancayo persiste una forma de economía que desafía la lógica capitalista.”],
[Claim],
[Standpoint about the alternative economic logic.],
[“El ’ayni’ —sistema de reciprocidad ancestral— organiza el intercambio entre vendedores y compradores.”],
[Premise],
[Evidence explaining the claim.],
[“Doña Juana no solo vende papas; participa en una red de solidaridad donde el dinero es solo una forma de intercambio entre muchas otras.”],
[Premise],
[Illustrative example supporting the claim.],
[“'Aquí nos ayudamos entre todos. Si alguien necesita, prestamos sin interés', explica.”],
[Premise],
[Example of collective support in practice.],
[“Estos espacios informales no buscan maximizar ganancias sino garantizar la subsistencia colectiva.”],
[Claim],
[Standpoint about the purpose of these markets.],
[“El antropólogo Carlos Montes señala: 'No es informalidad por falta de oportunidades, es una elección cultural que mantiene vivas tradiciones milenarias de cooperación'.”],
[Claim],
[Expert evaluative claim about cultural choice.],
[“Frente a la crisis del modelo neoliberal, estas economías alternativas ofrecen lecciones de sostenibilidad.”],
[Claim],
[Standpoint / conclusion about sustainability lessons.],
)

== Example 4
- #underline[Title:] “Informalidad laboral: herencia del neoliberalismo que precariza el trabajo”
- #underline[Content:] “El 75 % de trabajadores peruanos carece de seguridad social. Esta no es casualidad sino consecuencia directa de las reformas neoliberales de los 90 que flexibilizaron el mercado laboral. Las grandes empresas formales subcontratan servicios a microempresas informales, externalizando costos laborales. Juan trabaja 12 horas diarias repartiendo para una aplicación de delivery sin contrato ni beneficios. ’La empresa dice que soy independiente, pero ellos controlan todo’, lamenta. La economista Rosa Chávez explica: ’La informalidad no es disfunción del sistema sino su funcionamiento normal. El capital necesita este ejército de reserva precarizado para mantener salarios bajos’. Mientras persista este modelo, la formalización seguirá siendo un espejismo.”

#table(
  columns: 3,
  align: center+horizon,
  stroke: black,
[Sentence],
[Annotation],
[Justification],
[*Title:* “Informalidad laboral: herencia del neoliberalismo que precariza el trabajo”],
[Claim],
[Expresses evaluative standpoint on the neoliberal legacy.],
[“El 75\% de trabajadores peruanos carece de seguridad social.”],
[Premise],
[Evidence supporting claim about precarious work.],
[“Esta no es casualidad sino consecuencia directa de las reformas neoliberales de los 90 que flexibilizaron el mercado laboral.”],
[Claim],
[Standpoint linking precarity to structural reforms.],
[“Las grandes empresas formales subcontratan servicios a microempresas informales, externalizando costos laborales.”],
[Premise],
[Evidence explaining how system reproduces precarity.],
[“Juan trabaja 12 horas diarias repartiendo para una aplicaci\'on de delivery sin contrato ni beneficios.”],
[Premise],
[Example illustrating exploitative conditions.],
[“'La empresa dice que soy independiente, pero ellos controlan todo', lamenta.”],
[Premise],
[Evidence of labor control despite informal labeling.],
[“La economista Rosa Chávez explica: 'La informalidad no es disfunción del sistema sino su funcionamiento normal. El capital necesita este ejército de reserva precarizado para mantener salarios bajos'.”],
[Claim],
[Expert evaluative statement about systemic function.],
[“Mientras persista este modelo, la formalización seguiría siendo un espejismo.”],
[Claim],
[Predictive conclusion derived from prior premises.],
)

== Example 5
- #underline[Title:] “Construcción civil: el sector donde la informalidad es regla y no excepción”
- #underline[Content:] “En Lima se construyen miles de viviendas anualmente. La mayoría de trabajadores opera sin contrato formal. Luis, albañil con 20 años de experiencia, gana 80 soles diarios sin beneficios. Las constructoras contratan maestros de obra que subcontratan cuadrillas informales. Esta cadena reduce costos y traslada riesgos a los trabajadores. Cuando hay accidentes, las empresas niegan responsabilidad. El sector construcción concentra el 60 % de la informalidad laboral urbana. Los sindicatos denuncian que la fiscalización es insuficiente y las multas irrisorias.”

#table(
  columns: 3,
  align: center+horizon,
  stroke: black,
[Sentence],
[Annotation],
[Justification],
[*Title:* “Construcción civil: el sector donde la informalidad es regla y no excepción”],
[Claim],
[Expresses standpoint that informality dominates the construction sector.],
[“En Lima se construyen miles de viviendas anualmente.”],
[Premise],
[Contextual fact supporting the discussion.],
[“La mayoría de trabajadores opera sin contrato formal.”],
[Premise],
[Evidence of widespread informal employment.],
[“Luis, albañil con 20 años de experiencia, gana 80 soles diarios sin beneficios.”],
[Premise],
[Example illustrating precarious conditions.],
[“Las constructoras contratan maestros de obra que subcontratan cuadrillas informales.”],
[Premise],
[Evidence explaining structural organization.],
[“Esta cadena reduce costos y traslada riesgos a los trabajadores.”],
[Claim],
[Standpoint about exploitation in subcontracting.],
[“Cuando hay accidentes, las empresas niegan responsabilidad.”],
[Premise],
[Evidence supporting claim about risk transfer.],
[“El sector construcción concentra el 60\% de la informalidad laboral urbana.”],
[Premise],
[Statistical evidence supporting exploitation claim.],
[“Los sindicatos denuncian que la fiscalización es insuficiente y las multas irrisorias.”],
[Claim],
[Standpoint evaluating regulatory failure.],
)

= Consistency Guidelines
+ Annotate based on *function*, not grammatical form.
+ Do not infer intentions that are not *reasonably* supported by the article.
+ In cases of ambiguity, *prefer* `claim` if the segment expresses a clear position.