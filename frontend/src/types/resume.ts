export interface ExperienceItem {
  job_title: string;
  company: string;
  location?: string | null;
  start_date: string;
  end_date: string;
  bullets: string[];
}

export interface EducationItem {
  degree: string;
  institution: string;
  location?: string | null;
  graduation_date: string;
  details?: string[] | null;
}

export interface CertificationItem {
  name: string;
  issuer?: string | null;
  date?: string | null;
}

export interface ResumeData {
  name: string;
  email?: string | null;
  phone?: string | null;
  linkedin?: string | null;
  github?: string | null;
  location?: string | null;
  summary?: string | null;
  skills: string[];
  experience: ExperienceItem[];
  education: EducationItem[];
  certifications: CertificationItem[];
  ats_score_before?: number | null;
  ats_score?: number | null;
  matched_keywords?: string[];
}
