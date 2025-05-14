import React, { useState } from 'react';

export default function PatientForm({ onResult }) {
  const initial = { age: '', gender: '0', A1_Score: '', A2_Score: '', A3_Score: '', A4_Score: '', A5_Score: '', A6_Score: '', A7_Score: '', A8_Score: '', A9_Score: '', A10_Score: '' };
  const [form, setForm] = useState(initial);

  const handleChange = e => setForm({ ...form, [e.target.name]: e.target.value });

  const handleSubmit = async e => {
    e.preventDefault();
    const resp = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...form, age: +form.age, gender: +form.gender, ...Object.fromEntries(Object.entries(form).filter(([k]) => k.startsWith('A')).map(([k,v]) => [k, +v])) })
    });
    const data = await resp.json();
    onResult(data);
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>Age:<input name="age" value={form.age} onChange={handleChange} /></label>
      <label>Gender:<select name="gender" value={form.gender} onChange={handleChange}><option value="0">F</option><option value="1">M</option></select></label>
      {[...Array(10)].map((_, i) => (
        <label key={i}>Q{i+1}:<input name={`A${i+1}_Score`} value={form[`A${i+1}_Score`]} onChange={handleChange} /></label>
      ))}
      <button type="submit">Submit</button>
    </form>
  );
}