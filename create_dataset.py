"""
create_dataset.py — Genera 250+ documentos de CVs e iş ilanları
"""
import pandas as pd
import random
import os

random.seed(42)

kategoriler = [
    "Data Science", "Software Engineering", "Marketing", "Finance",
    "Human Resources", "Healthcare", "Education", "Sales",
    "Project Management", "Cybersecurity"
]

yetenekler = {
    "Data Science": [
        "python machine learning deep learning neural networks tensorflow pytorch scikit-learn pandas numpy data analysis statistical modeling regression classification clustering natural language processing computer vision data visualization matplotlib seaborn sql database big data spark hadoop feature engineering model deployment mlops",
        "statistical analysis predictive modeling data mining exploratory data analysis hypothesis testing ab testing time series forecasting business intelligence tableau power bi r programming data warehousing etl pipeline cloud computing aws gcp azure jupyter notebook git version control",
        "supervised learning unsupervised learning reinforcement learning convolutional neural networks recurrent neural networks transformers bert gpt word embeddings topic modeling sentiment analysis named entity recognition text classification image recognition object detection",
        "data engineering apache spark kafka airflow dbt snowflake databricks data lake data mesh feature store mlflow kubeflow vertex ai sagemaker data quality monitoring model drift detection explainable ai responsible ai causal inference bayesian methods",
    ],
    "Software Engineering": [
        "java python javascript typescript react angular vue nodejs express django flask spring boot microservices rest api graphql docker kubernetes devops ci cd git agile scrum software architecture design patterns object oriented programming functional programming",
        "backend development frontend development full stack web development mobile development ios android react native flutter database design postgresql mysql mongodb redis elasticsearch message queues kafka rabbitmq cloud services aws azure google cloud serverless functions",
        "software testing unit testing integration testing test driven development code review refactoring performance optimization security best practices authentication authorization oauth jwt api design documentation technical writing system design distributed systems",
        "golang rust scala elixir phoenix event driven architecture cqrs event sourcing domain driven design hexagonal architecture clean code solid principles twelve factor app observability logging monitoring tracing prometheus grafana datadog",
    ],
    "Marketing": [
        "digital marketing social media marketing content marketing email marketing search engine optimization seo search engine marketing sem google ads facebook ads instagram marketing brand management marketing strategy customer acquisition lead generation conversion optimization",
        "market research competitive analysis customer segmentation buyer persona campaign management marketing automation hubspot salesforce crm analytics google analytics content creation copywriting blog writing social media management community management influencer marketing",
        "product marketing go to market strategy launch campaigns brand awareness demand generation inbound marketing outbound marketing account based marketing public relations media relations press releases event marketing trade shows webinars customer retention loyalty programs",
        "growth hacking viral marketing referral programs affiliate marketing programmatic advertising retargeting lookalike audiences customer lifetime value cohort analysis funnel optimization landing page optimization ab testing multivariate testing attribution modeling",
    ],
    "Finance": [
        "financial analysis financial modeling budgeting forecasting accounting financial reporting variance analysis cost analysis profit loss balance sheet cash flow statement excel advanced formulas pivot tables vlookup financial planning investment analysis portfolio management",
        "corporate finance valuation discounted cash flow dcf mergers acquisitions due diligence risk management compliance regulatory reporting gaap ifrs auditing internal controls sox compliance tax planning treasury management working capital management",
        "quantitative analysis derivatives options futures fixed income equity research bloomberg terminal reuters trading financial markets investment banking private equity venture capital hedge funds mutual funds asset management wealth management financial advisory",
        "crypto defi blockchain fintech payments processing banking regulation basel iii stress testing liquidity risk credit risk market risk operational risk model risk management esg investing sustainable finance impact investing factor investing smart beta",
    ],
    "Human Resources": [
        "talent acquisition recruitment sourcing interviewing candidate assessment onboarding employee relations performance management compensation benefits administration hr policies procedures employee engagement retention training development succession planning workforce planning",
        "human resources information systems hris workday sap hr oracle hr applicant tracking system ats linkedin recruiter boolean search behavioral interviewing competency based hiring diversity inclusion equity belonging organizational development change management",
        "labor relations employment law compliance payroll administration leave management conflict resolution disciplinary procedures termination offboarding hr analytics people analytics employee surveys culture building team building leadership development coaching mentoring",
        "total rewards job architecture salary benchmarking equity compensation stock options restricted stock units executive compensation benefits design wellness programs employee experience design employer branding talent brand candidate experience glassdoor management",
    ],
    "Healthcare": [
        "patient care clinical documentation electronic health records ehr epic cerner medical terminology anatomy physiology pharmacology nursing diagnosis treatment planning medication administration vital signs monitoring patient assessment healthcare compliance hipaa",
        "medical coding billing icd cpt healthcare administration hospital operations quality improvement patient safety infection control evidence based practice clinical research biostatistics public health epidemiology health policy healthcare informatics telemedicine",
        "physical therapy occupational therapy speech therapy rehabilitation patient education wound care surgical procedures operating room intensive care unit emergency medicine pediatrics geriatrics oncology cardiology neurology orthopedics mental health psychiatry",
        "population health management chronic disease management care coordination case management utilization review prior authorization denial management revenue cycle management value based care accountable care organization patient centered medical home",
    ],
    "Education": [
        "curriculum development lesson planning classroom management student assessment differentiated instruction special education english language learners student engagement active learning cooperative learning project based learning technology integration",
        "educational psychology child development learning theories behavior management parent communication school counseling academic advising college preparation standardized testing tutoring mentoring student support services inclusive education universal design learning",
        "higher education academic research teaching university college faculty instruction course design online learning lms canvas blackboard moodle academic publishing peer review grant writing research methodology literature review thesis dissertation advising",
        "instructional design elearning development articulate storyline adobe captivate learning management systems training needs analysis performance consulting blended learning microlearning gamification badges certifications corporate training workforce development",
    ],
    "Sales": [
        "b2b sales b2c sales account management business development cold calling prospecting lead qualification sales pipeline crm salesforce hubspot sales presentations negotiation closing techniques customer relationship management territory management quota achievement",
        "solution selling consultative selling value based selling enterprise sales inside sales outside sales field sales channel sales partner management distributor relationships revenue growth market penetration customer success upselling cross selling renewal management",
        "sales strategy sales forecasting sales analytics competitive intelligence product knowledge technical sales demo presentations proof of concept proposal writing contract negotiation sales operations sales enablement sales training coaching team leadership",
        "saas sales subscription sales recurring revenue arr mrr churn reduction expansion revenue land and expand outbound prospecting sequence automation outreach apollo zoominfo linkedin sales navigator social selling video prospecting",
    ],
    "Project Management": [
        "project planning project scheduling resource allocation budget management risk management stakeholder management project documentation status reporting change management project closure lessons learned pmp certification agile scrum kanban waterfall methodology",
        "microsoft project jira confluence trello asana monday gantt charts work breakdown structure critical path method earned value management project governance program management portfolio management benefits realization strategic alignment cross functional teams",
        "scope management requirements gathering business analysis process improvement lean six sigma quality management vendor management contract management procurement outsourcing offshore teams remote team management virtual collaboration communication facilitation",
        "okr goal setting strategic planning roadmap development product roadmap technology roadmap digital transformation innovation management design thinking human centered design service design process redesign automation robotic process automation rpa",
    ],
    "Cybersecurity": [
        "network security information security penetration testing vulnerability assessment ethical hacking firewall intrusion detection prevention systems siem endpoint protection antivirus malware analysis threat intelligence",
        "security operations center incident response digital forensics log analysis security monitoring network traffic analysis wireshark nmap metasploit kali linux security auditing compliance pci dss iso 27001 nist framework zero trust",
        "cloud security application security web application security owasp devsecops secure coding code review static analysis dynamic analysis authentication authorization identity access management encryption cryptography",
        "threat hunting red team blue team purple team adversary simulation mitre att ck framework threat modeling security architecture security engineering bug bounty responsible disclosure cve vulnerability research exploit development",
    ],
}

deneyimler = {
    "Data Science": [
        "developed machine learning models to predict customer churn reducing attrition by 25 percent built recommendation system increasing user engagement by 40 percent created automated data pipeline processing 10 million records daily",
        "analyzed large datasets using python and sql to identify business opportunities led team of 5 data scientists delivering projects on time published research paper on deep learning applications",
        "designed and deployed natural language processing system for automated customer support handling 1000 queries daily built real time fraud detection model saving company 2 million dollars annually",
        "built computer vision system detecting product defects with 99 percent accuracy reducing quality control costs by 60 percent implemented ab testing framework running 50 simultaneous experiments",
        "created demand forecasting models reducing inventory costs by 30 percent developed customer segmentation analysis identifying 5 high value segments built executive dashboard used by c suite daily",
    ],
    "Software Engineering": [
        "developed scalable microservices architecture serving 5 million users daily reduced api response time by 60 percent led migration from monolithic to cloud native architecture saving 30 percent infrastructure costs",
        "built full stack web application used by 100000 customers implemented ci cd pipeline reducing deployment time from days to hours mentored junior developers maintaining high code quality",
        "designed distributed system processing 1 billion events per day contributed to open source projects optimized database queries reducing load time by 80 percent",
        "led technical architecture review modernizing legacy codebase reducing technical debt by 50 percent built developer platform increasing team productivity by 35 percent",
        "created mobile application with 500000 downloads implemented offline first architecture improving user experience in low connectivity environments reduced crash rate to below 0.1 percent",
    ],
    "Marketing": [
        "managed social media accounts growing followers from 10000 to 500000 increased email open rates by 45 percent led product launch campaign generating 5 million dollars in first month revenue",
        "developed content strategy increasing organic traffic by 300 percent managed google ads campaigns with 5 million dollar annual budget achieving 3x roi created brand guidelines adopted across 20 countries",
        "built marketing automation system reducing manual work by 70 percent conducted market research informing product roadmap established influencer partnerships reaching 10 million potential customers",
        "launched referral program acquiring 50000 new customers at 10 percent of traditional cac implemented personalization engine increasing conversion rates by 25 percent",
        "redesigned company website improving bounce rate by 40 percent and increasing time on site by 3 minutes ran 100 ab tests per quarter systematically improving all key metrics",
    ],
    "Finance": [
        "prepared financial models supporting 500 million dollar acquisition decision managed 50 million dollar budget identifying 3 million in cost savings implemented financial reporting system reducing close time by 5 days",
        "led team of 8 analysts producing quarterly financial reports developed risk management framework adopted company wide identified investment opportunities generating 20 percent annual return",
        "managed investment portfolio of 100 million dollars achieving 15 percent above benchmark conducted due diligence on 30 potential acquisition targets built financial planning model for 5 year strategy",
        "restructured 200 million dollar debt facility saving 10 million in annual interest expense implemented treasury management system improving cash visibility across 15 countries",
        "built financial model for ipo roadshow raising 500 million dollars developed pricing model for new product line improving gross margin by 8 percentage points",
    ],
    "Human Resources": [
        "reduced time to hire from 60 to 30 days hired 200 employees in one year maintaining 90 percent retention designed onboarding program increasing new hire productivity by 40 percent",
        "implemented performance management system for 1000 employees managing compensation review saving company 5 million annually developed training programs reducing turnover by 35 percent",
        "led diversity initiative increasing diverse hires by 50 percent redesigned benefits package improving employee satisfaction from 65 to 85 percent",
        "built people analytics function providing data driven insights to business leaders implemented predictive attrition model allowing proactive retention interventions",
        "negotiated new union contract maintaining positive labor relations implemented hris system streamlining hr processes saving 2000 hours of manual work annually",
    ],
    "Healthcare": [
        "provided patient care to 20 patients daily maintaining 98 percent satisfaction implemented clinical protocol reducing hospital acquired infections by 40 percent trained 15 new staff",
        "managed electronic health records implementation across 5 departments improving documentation accuracy by 60 percent coordinated care reducing readmission rates by 25 percent",
        "conducted clinical research resulting in 3 peer reviewed publications developed patient education materials improved medication adherence by 30 percent",
        "implemented telehealth program serving 500 patients monthly during pandemic maintaining care continuity developed triage protocol reducing emergency department wait times by 30 percent",
        "led quality improvement initiative achieving joint commission accreditation without deficiencies developed staffing model optimizing nurse to patient ratios reducing burnout",
    ],
    "Education": [
        "developed curriculum improving student test scores by 30 percent implemented technology integration across 20 classrooms mentored 5 first year teachers improving their effectiveness",
        "designed online learning platform used by 5000 students conducted action research published in educational journal created inclusive classroom achieving 95 percent student engagement",
        "established tutoring program improving at risk student graduation rate by 20 percent coordinated professional development for 50 teachers",
        "launched project based learning initiative increasing student motivation scores by 40 percent created community partnership program providing real world learning experiences",
        "implemented social emotional learning curriculum reducing disciplinary incidents by 50 percent developed family engagement program increasing parent participation by 80 percent",
    ],
    "Sales": [
        "exceeded sales quota by 150 percent for 3 consecutive years generating 5 million dollars in new revenue built territory from zero to 2 million in annual recurring revenue",
        "led sales team of 10 representatives achieving 120 percent of team quota implemented new methodology increasing conversion rate by 35 percent",
        "closed largest deal in company history valued at 10 million dollars expanded existing account growing annual contract from 500000 to 2 million",
        "built enterprise sales motion from scratch closing first 5 fortune 500 customers created scalable outbound process generating 2 million in qualified pipeline monthly",
        "implemented customer success program reducing churn by 40 percent and increasing net revenue retention to 130 percent expanded regional territory generating 8 million in new revenue",
    ],
    "Project Management": [
        "delivered 15 million dollar erp implementation on time and under budget managed cross functional team of 30 people across 3 countries implemented agile reducing delivery time by 40 percent",
        "led digital transformation initiative modernizing legacy systems affecting 5000 employees managed vendor relationships for 10 million dollar technology contract",
        "delivered complex infrastructure project 2 months ahead of schedule saving 1 million created risk framework preventing 3 major project failures",
        "managed portfolio of 20 concurrent projects totaling 50 million dollars establishing pmo that improved on time delivery from 60 to 90 percent",
        "led organizational change management for major restructuring affecting 2000 employees achieving 85 percent adoption within 6 months of new systems",
    ],
    "Cybersecurity": [
        "led incident response team handling 500 security incidents annually preventing estimated 10 million dollars in damages implemented zero trust architecture reducing breaches by 70 percent",
        "conducted penetration testing identifying 200 critical vulnerabilities built security operations center monitoring 50000 endpoints 24 hours 7 days",
        "developed security training reducing phishing success rate from 30 to 5 percent implemented cloud security controls achieving iso 27001 certification",
        "built threat intelligence program identifying and neutralizing 3 advanced persistent threat groups conducted red team exercises exposing critical vulnerabilities in production systems",
        "led security architecture for cloud migration ensuring zero security incidents designed identity access management system for 10000 employees reducing insider threat risk by 80 percent",
    ],
}

eğitim = {
    "Data Science": ["bachelor degree computer science statistics mathematics", "master degree data science machine learning artificial intelligence", "phd computational linguistics natural language processing", "master degree applied mathematics statistics university"],
    "Software Engineering": ["bachelor degree computer science software engineering", "master degree computer science software systems", "bachelor degree electrical engineering computer science", "self taught software engineer bootcamp coding"],
    "Marketing": ["bachelor degree marketing communications business administration", "master degree marketing digital marketing business", "bachelor degree journalism communications public relations", "master degree business administration mba marketing"],
    "Finance": ["bachelor degree finance accounting economics", "master degree finance mba business administration", "chartered financial analyst cfa certified public accountant cpa", "master degree financial engineering quantitative finance"],
    "Human Resources": ["bachelor degree human resources management psychology", "master degree human resources organizational behavior", "professional human resources phr senior professional human resources sphr", "master degree business administration organizational development"],
    "Healthcare": ["bachelor degree nursing registered nurse", "master degree nursing healthcare administration public health", "doctor medicine physician medical school residency", "bachelor degree health sciences pre-medicine biology"],
    "Education": ["bachelor degree education teaching elementary secondary", "master degree education curriculum instruction educational leadership", "doctor education edd phd educational psychology", "master degree special education learning disabilities"],
    "Sales": ["bachelor degree business administration sales marketing", "master degree business administration mba sales management", "bachelor degree communications psychology", "bachelor degree liberal arts business minor"],
    "Project Management": ["bachelor degree business administration project management", "master degree project management mba operations management", "project management professional pmp agile certified practitioner", "bachelor degree engineering industrial management"],
    "Cybersecurity": ["bachelor degree computer science information security cybersecurity", "master degree information security cybersecurity network security", "certified information systems security professional cissp certified ethical hacker ceh", "bachelor degree information technology security administration"],
}


temel_adlar = [
    "James", "Sarah", "Michael", "Emily", "David", "Jessica", "Daniel", "Ashley",
    "Christopher", "Amanda", "Matthew", "Stephanie", "Joshua", "Megan", "Andrew",
    "Lauren", "Ryan", "Kayla", "Brandon", "Rachel", "Kevin", "Brittany", "Justin",
    "Samantha", "Eric", "Melissa", "Timothy", "Nicole", "Jeremy", "Amber",
    "Marcus", "Vanessa", "Derrick", "Monica", "Troy", "Jasmine", "Carlos", "Sofia",
    "Miguel", "Isabella", "Luis", "Valentina", "Ahmed", "Fatima", "Omar", "Layla",
    "Chen", "Wei", "Yuki", "Akira", "Raj", "Priya", "Arjun", "Neha",
]
apellidos = [
    "Wilson", "Johnson", "Brown", "Davis", "Martinez", "Taylor", "Anderson",
    "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia",
    "Robinson", "Lewis", "Walker", "Hall", "Allen", "Young", "Hernandez",
    "King", "Wright", "Lopez", "Hill", "Scott", "Green", "Adams", "Baker",
    "Nelson", "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips",
    "Campbell", "Parker", "Evans", "Edwards", "Collins", "Stewart", "Sanchez",
    "Morris", "Rogers", "Reed", "Cook", "Morgan", "Bell", "Murphy",
]

registros = []
doc_id = 1
random.shuffle(temel_adlar)

for i in range(250):
    categoria = kategoriler[i % len(kategoriler)]
    nombre = f"{temel_adlar[i % len(temel_adlar)]} {apellidos[i % len(apellidos)]}"
    anos = random.randint(1, 18)
    
    hab = random.choice(yetenekler[categoria])
    exp = random.choice(deneyimler[categoria])
    edu = random.choice(eğitim[categoria])
    
    nivel = random.choice(["junior", "mid level", "senior", "lead", "principal", "staff"])
    
    cv = f"""
{nombre} {nivel} professional with {anos} years experience in {categoria.lower()} field.

SKILLS AND EXPERTISE:
{hab}

PROFESSIONAL EXPERIENCE:
{exp}

EDUCATION:
{edu}

PROFESSIONAL SUMMARY:
Results driven {categoria} professional with {anos} years of progressive experience.
Proven ability to deliver results collaborate effectively and drive continuous improvement.
Strong communication leadership and analytical skills with passion for innovation.
    """.strip()
    
    registros.append({
        "document_id": f"cv_{doc_id:03d}",
        "content": cv,
        "category": categoria,
        "type": "cv"
    })
    doc_id += 1

# İş ilanları
job_templates = [
    ("Data Science", "We are seeking experienced Data Scientist to join our team. Requirements include python machine learning deep learning tensorflow pytorch scikit-learn pandas numpy statistical modeling regression classification clustering natural language processing data visualization sql big data spark feature engineering model deployment mlops cloud computing. Responsibilities include developing machine learning models analyzing large datasets building data pipelines presenting findings to stakeholders. Minimum 5 years experience required master degree preferred."),
    ("Software Engineering", "Full Stack Software Engineer needed. Requirements include javascript typescript react nodejs python django rest api docker kubernetes microservices postgresql mongodb redis git devops ci cd cloud services. Responsibilities include developing web applications building scalable backend systems conducting code reviews collaborating with product team. Minimum 3 years experience bachelor degree computer science required."),
    ("Marketing", "Digital Marketing Manager position available. Requirements include digital marketing social media seo sem google ads facebook ads content marketing email marketing automation hubspot salesforce crm analytics brand management. Responsibilities include managing digital campaigns developing content strategy growing social media presence analyzing performance. Minimum 4 years experience required."),
    ("Finance", "Financial Analyst role at growing firm. Requirements include financial modeling excel accounting financial reporting analysis budgeting forecasting bloomberg financial markets investment analysis portfolio management valuation. Responsibilities include preparing financial models analyzing opportunities presenting to management monitoring performance. Minimum 3 years experience required."),
    ("Human Resources", "HR Business Partner needed. Requirements include talent acquisition recruitment performance management compensation benefits hris workday employee relations hr policies compliance training development workforce planning. Responsibilities include partnering with business leaders managing recruitment handling employee relations ensuring compliance. Minimum 5 years experience required."),
    ("Healthcare", "Clinical Professional position open. Requirements include patient care clinical documentation electronic health records ehr medical terminology pharmacology nursing diagnosis treatment planning medication administration patient assessment hipaa compliance. Responsibilities include providing patient care maintaining documentation coordinating with medical team. License and certification required."),
    ("Education", "Educator position available. Requirements include curriculum development lesson planning classroom management student assessment differentiated instruction technology integration educational psychology learning theories behavior management. Responsibilities include developing curriculum delivering instruction assessing student progress. Teaching certification required."),
    ("Sales", "Account Executive position. Requirements include b2b sales account management business development prospecting lead qualification sales pipeline crm salesforce presentations negotiation closing customer relationship management. Responsibilities include managing accounts developing new business presenting solutions. Minimum 3 years sales experience quota achievement required."),
    ("Project Management", "Project Manager needed. Requirements include project planning scheduling resource allocation budget management risk management stakeholder management agile scrum kanban pmp certification microsoft project jira. Responsibilities include leading projects managing teams ensuring delivery within scope budget and schedule. PMP preferred minimum 5 years experience."),
    ("Cybersecurity", "Cybersecurity Engineer position. Requirements include network security penetration testing vulnerability assessment ethical hacking firewall intrusion detection siem endpoint protection malware analysis threat intelligence cloud security application security. Responsibilities include monitoring security systems responding to incidents conducting assessments. CISSP or CEH preferred."),
]

for j, (cat, desc) in enumerate(job_templates):
    for k in range(3):
        nivel_job = random.choice(["Junior", "Mid-Level", "Senior", "Lead"])
        registros.append({
            "document_id": f"job_{j*3+k+1:03d}",
            "content": f"{nivel_job} {cat} position at leading company. {desc}",
            "category": cat,
            "type": "job_posting"
        })

df = pd.DataFrame(registros)
df_model = df[["document_id", "content"]].copy()

os.makedirs('/mnt/user-data/outputs/cv_matching/data', exist_ok=True)
df.to_csv('/mnt/user-data/outputs/cv_matching/data/cv_jobs_raw.csv', index=False)
df_model.to_csv('/mnt/user-data/outputs/cv_matching/data/cv_jobs_dataset.csv', index=False)

print(f"✅ Dataset generado:")
print(f"   Total documentos: {len(df)}")
print(f"   CVs: {len(df[df['type']=='cv'])}")
print(f"   Job postings: {len(df[df['type']=='job_posting'])}")
print(f"   Categorías: {df['category'].nunique()}")
print(f"   Palabras promedio por documento: {df['content'].apply(lambda x: len(x.split())).mean():.0f}")
